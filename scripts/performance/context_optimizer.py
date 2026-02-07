#!/usr/bin/env python3
"""
context_optimizer.py

Context Window Optimizer for USB-AI.
Optimizes context length and KV cache for faster inference latency.

Key optimizations:
- Dynamic context sizing based on use case
- Sliding window context for long conversations
- KV cache configuration based on available memory
- Token estimation and prompt efficiency
- Automatic context compression triggers
"""

import json
import logging
import os
import platform
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from enum import Enum

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

__version__ = "1.0.0"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)


class ContextProfile(Enum):
    """Predefined context profiles for different use cases."""
    QUICK_RESPONSE = "quick_response"
    STANDARD = "standard"
    EXTENDED = "extended"
    MAXIMUM = "maximum"
    CUSTOM = "custom"


@dataclass
class ContextConfig:
    """Context window configuration."""
    num_ctx: int                      # Context window size (tokens)
    num_batch: int                    # Batch size for prompt processing
    num_predict: int                  # Max tokens to generate (-1 = unlimited)
    system_prompt_budget: int         # Tokens reserved for system prompt
    kv_cache_type: str               # KV cache quantization (f16, q8_0, q4_0)
    rope_frequency_base: float       # RoPE base frequency for extended context
    rope_frequency_scale: float      # RoPE scaling factor
    sliding_window: int              # Sliding window size (0 = disabled)
    sliding_window_overlap: int      # Overlap between windows

    def to_ollama_options(self) -> Dict[str, Any]:
        """Convert to Ollama API options format."""
        options = {
            "num_ctx": self.num_ctx,
            "num_batch": self.num_batch,
        }
        if self.num_predict > 0:
            options["num_predict"] = self.num_predict
        return options


@dataclass
class TokenEstimate:
    """Token count estimation for text."""
    text_length: int
    estimated_tokens: int
    chars_per_token: float
    method: str  # "tiktoken", "simple", "word_based"


@dataclass
class ContextOptimizationResult:
    """Result of context optimization analysis."""
    recommended_profile: ContextProfile
    config: ContextConfig
    estimated_latency_ms: float
    memory_usage_mb: float
    tokens_per_second_estimate: float
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)


class TokenEstimator:
    """Estimates token counts for text without requiring tokenizer libraries."""

    # Average characters per token for common model families
    MODEL_CPT = {
        "llama": 3.5,
        "dolphin": 3.5,
        "mistral": 3.8,
        "qwen": 3.2,
        "phi": 4.0,
        "gemma": 3.6,
        "codellama": 3.3,
        "deepseek": 3.4,
        "default": 3.5,
    }

    def __init__(self, model_family: str = "default"):
        """Initialize with model family for accurate estimation."""
        self.model_family = model_family.lower()
        self.chars_per_token = self._get_chars_per_token()

    def _get_chars_per_token(self) -> float:
        """Get characters per token ratio for model family."""
        for family, cpt in self.MODEL_CPT.items():
            if family in self.model_family:
                return cpt
        return self.MODEL_CPT["default"]

    def estimate(self, text: str) -> TokenEstimate:
        """Estimate token count for text."""
        text_length = len(text)

        if text_length == 0:
            return TokenEstimate(
                text_length=0,
                estimated_tokens=0,
                chars_per_token=self.chars_per_token,
                method="simple"
            )

        # Simple character-based estimation
        simple_estimate = int(text_length / self.chars_per_token)

        # Adjust for code content (more tokens due to special characters)
        code_indicators = ["```", "def ", "function ", "class ", "import ", "{", "}"]
        if any(indicator in text for indicator in code_indicators):
            simple_estimate = int(simple_estimate * 1.15)

        # Adjust for heavy punctuation
        punct_ratio = sum(1 for c in text if c in ".,;:!?()[]{}\"'") / max(text_length, 1)
        if punct_ratio > 0.1:
            simple_estimate = int(simple_estimate * (1 + punct_ratio * 0.5))

        return TokenEstimate(
            text_length=text_length,
            estimated_tokens=max(1, simple_estimate),
            chars_per_token=self.chars_per_token,
            method="simple"
        )

    def estimate_conversation(self, messages: List[Dict[str, str]]) -> int:
        """Estimate total tokens for a conversation."""
        total = 0
        for msg in messages:
            # Account for role prefix tokens (~4 tokens per message overhead)
            total += 4
            content = msg.get("content", "")
            total += self.estimate(content).estimated_tokens
        return total

    def estimate_remaining_context(
        self,
        current_tokens: int,
        max_context: int,
        reserve_for_response: int = 512
    ) -> int:
        """Calculate remaining context space for new input."""
        return max(0, max_context - current_tokens - reserve_for_response)


class MemoryAnalyzer:
    """Analyzes available memory for KV cache sizing."""

    # KV cache memory per token (approximate bytes)
    KV_BYTES_PER_TOKEN = {
        "f16": 512,      # 2 bytes per element, 256 elements typical
        "q8_0": 256,     # 1 byte per element
        "q4_0": 128,     # 0.5 bytes per element
    }

    def __init__(self):
        self.system = platform.system().lower()

    def get_available_memory_mb(self) -> float:
        """Get available system memory in MB."""
        try:
            if self.system == "linux":
                with open("/proc/meminfo") as f:
                    meminfo = f.read()
                match = re.search(r"MemAvailable:\s+(\d+)\s+kB", meminfo)
                if match:
                    return int(match.group(1)) / 1024

            elif self.system == "darwin":
                result = subprocess.run(
                    ["vm_stat"],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    free_match = re.search(r"Pages free:\s+(\d+)", result.stdout)
                    inactive_match = re.search(r"Pages inactive:\s+(\d+)", result.stdout)
                    free_pages = int(free_match.group(1)) if free_match else 0
                    inactive_pages = int(inactive_match.group(1)) if inactive_match else 0
                    return (free_pages + inactive_pages) * 4096 / 1024 / 1024

            elif self.system == "windows":
                result = subprocess.run(
                    ["wmic", "OS", "get", "FreePhysicalMemory"],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    lines = result.stdout.strip().split("\n")
                    if len(lines) > 1:
                        return int(lines[1].strip()) / 1024

        except Exception as e:
            log.warning(f"Memory detection fallback: {e}")

        return 4096.0  # Default 4GB assumption

    def calculate_kv_cache_memory(
        self,
        context_length: int,
        num_layers: int = 32,
        kv_type: str = "f16"
    ) -> float:
        """Calculate KV cache memory requirement in MB."""
        bytes_per_token = self.KV_BYTES_PER_TOKEN.get(kv_type, 512)
        # Each layer has key and value caches
        total_bytes = context_length * num_layers * 2 * bytes_per_token
        return total_bytes / 1024 / 1024

    def recommend_kv_cache_type(self, available_mb: float, context_length: int) -> str:
        """Recommend KV cache type based on available memory."""
        f16_requirement = self.calculate_kv_cache_memory(context_length, kv_type="f16")
        q8_requirement = self.calculate_kv_cache_memory(context_length, kv_type="q8_0")
        q4_requirement = self.calculate_kv_cache_memory(context_length, kv_type="q4_0")

        # Leave 30% memory for model weights and other operations
        usable_memory = available_mb * 0.7

        if f16_requirement <= usable_memory:
            return "f16"
        elif q8_requirement <= usable_memory:
            return "q8_0"
        elif q4_requirement <= usable_memory:
            return "q4_0"
        else:
            return "q4_0"  # Fallback to smallest

    def max_context_for_memory(
        self,
        available_mb: float,
        kv_type: str = "f16",
        num_layers: int = 32
    ) -> int:
        """Calculate maximum context length for available memory."""
        bytes_per_token = self.KV_BYTES_PER_TOKEN.get(kv_type, 512)
        usable_mb = available_mb * 0.7
        usable_bytes = usable_mb * 1024 * 1024
        max_tokens = int(usable_bytes / (num_layers * 2 * bytes_per_token))
        # Round down to nearest power of 2 for efficiency
        power = 1
        while power * 2 <= max_tokens:
            power *= 2
        return min(power, 131072)  # Cap at 128K


class SlidingWindowManager:
    """Manages sliding window context for long conversations."""

    def __init__(
        self,
        window_size: int = 4096,
        overlap: int = 512,
        summarize_threshold: float = 0.8
    ):
        """
        Initialize sliding window manager.

        Args:
            window_size: Maximum tokens in active window
            overlap: Tokens to retain when sliding
            summarize_threshold: Trigger summarization at this fill ratio
        """
        self.window_size = window_size
        self.overlap = overlap
        self.summarize_threshold = summarize_threshold
        self.token_estimator = TokenEstimator()

    def should_slide(self, current_tokens: int) -> bool:
        """Check if window needs to slide."""
        return current_tokens >= int(self.window_size * self.summarize_threshold)

    def calculate_retention(
        self,
        messages: List[Dict[str, str]],
        keep_system: bool = True
    ) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        """
        Calculate which messages to retain and which to summarize.

        Returns:
            Tuple of (messages_to_keep, messages_to_summarize)
        """
        if not messages:
            return [], []

        retained = []
        to_summarize = []
        current_tokens = 0

        # Always keep system message if present
        if keep_system and messages[0].get("role") == "system":
            retained.append(messages[0])
            current_tokens += self.token_estimator.estimate(
                messages[0].get("content", "")
            ).estimated_tokens + 4
            messages = messages[1:]

        # Process messages from newest to oldest
        for msg in reversed(messages):
            msg_tokens = self.token_estimator.estimate(
                msg.get("content", "")
            ).estimated_tokens + 4

            if current_tokens + msg_tokens <= self.overlap:
                retained.insert(0 if not retained or retained[0].get("role") != "system" else 1, msg)
                current_tokens += msg_tokens
            else:
                to_summarize.insert(0, msg)

        return retained, to_summarize

    def create_summary_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Create a prompt for summarizing old messages."""
        if not messages:
            return ""

        conversation_text = []
        for msg in messages:
            role = msg.get("role", "user").capitalize()
            content = msg.get("content", "")
            conversation_text.append(f"{role}: {content}")

        return f"""Summarize the following conversation concisely, preserving key facts, decisions, and context:

{chr(10).join(conversation_text)}

Summary:"""


class ContextOptimizer:
    """Main optimizer for context window and KV cache configuration."""

    # Profile definitions
    PROFILES = {
        ContextProfile.QUICK_RESPONSE: {
            "num_ctx": 512,
            "num_batch": 128,
            "num_predict": 256,
            "system_prompt_budget": 128,
            "kv_cache_type": "q4_0",
            "rope_frequency_base": 10000.0,
            "rope_frequency_scale": 1.0,
            "sliding_window": 0,
            "sliding_window_overlap": 0,
            "description": "Ultra-fast responses, minimal context",
            "use_cases": ["quick Q&A", "simple commands", "status checks"],
            "latency_target_ms": 50,
        },
        ContextProfile.STANDARD: {
            "num_ctx": 2048,
            "num_batch": 256,
            "num_predict": 512,
            "system_prompt_budget": 256,
            "kv_cache_type": "q8_0",
            "rope_frequency_base": 10000.0,
            "rope_frequency_scale": 1.0,
            "sliding_window": 0,
            "sliding_window_overlap": 0,
            "description": "Balanced speed and context",
            "use_cases": ["general chat", "short coding tasks", "explanations"],
            "latency_target_ms": 100,
        },
        ContextProfile.EXTENDED: {
            "num_ctx": 4096,
            "num_batch": 512,
            "num_predict": 1024,
            "system_prompt_budget": 512,
            "kv_cache_type": "f16",
            "rope_frequency_base": 10000.0,
            "rope_frequency_scale": 1.0,
            "sliding_window": 4096,
            "sliding_window_overlap": 1024,
            "description": "Extended context for longer conversations",
            "use_cases": ["multi-turn conversations", "code review", "document analysis"],
            "latency_target_ms": 200,
        },
        ContextProfile.MAXIMUM: {
            "num_ctx": 8192,
            "num_batch": 1024,
            "num_predict": 2048,
            "system_prompt_budget": 1024,
            "kv_cache_type": "f16",
            "rope_frequency_base": 10000.0,
            "rope_frequency_scale": 1.0,
            "sliding_window": 8192,
            "sliding_window_overlap": 2048,
            "description": "Maximum context for long documents",
            "use_cases": ["long documents", "complex analysis", "book summaries"],
            "latency_target_ms": 500,
        },
    }

    def __init__(self, model_family: str = "llama"):
        """Initialize context optimizer."""
        self.model_family = model_family
        self.token_estimator = TokenEstimator(model_family)
        self.memory_analyzer = MemoryAnalyzer()

    def get_profile_config(self, profile: ContextProfile) -> ContextConfig:
        """Get configuration for a profile."""
        if profile not in self.PROFILES:
            profile = ContextProfile.STANDARD

        p = self.PROFILES[profile]
        return ContextConfig(
            num_ctx=p["num_ctx"],
            num_batch=p["num_batch"],
            num_predict=p["num_predict"],
            system_prompt_budget=p["system_prompt_budget"],
            kv_cache_type=p["kv_cache_type"],
            rope_frequency_base=p["rope_frequency_base"],
            rope_frequency_scale=p["rope_frequency_scale"],
            sliding_window=p["sliding_window"],
            sliding_window_overlap=p["sliding_window_overlap"],
        )

    def analyze_use_case(
        self,
        prompt_length: int = 0,
        expected_response_length: int = 0,
        conversation_turns: int = 1,
        has_code: bool = False,
        has_documents: bool = False,
        priority: str = "balanced"  # "speed", "balanced", "quality"
    ) -> ContextProfile:
        """Analyze use case and recommend optimal profile."""

        # Quick response for very short interactions
        if prompt_length < 100 and expected_response_length < 200 and priority == "speed":
            return ContextProfile.QUICK_RESPONSE

        # Extended context for code or documents
        if has_code or has_documents:
            if prompt_length > 2000 or conversation_turns > 5:
                return ContextProfile.MAXIMUM
            return ContextProfile.EXTENDED

        # Multi-turn conversations need more context
        if conversation_turns > 10:
            return ContextProfile.MAXIMUM
        if conversation_turns > 5:
            return ContextProfile.EXTENDED

        # Priority-based selection
        if priority == "speed":
            if prompt_length < 500:
                return ContextProfile.QUICK_RESPONSE
            return ContextProfile.STANDARD
        elif priority == "quality":
            return ContextProfile.EXTENDED

        # Default balanced selection based on prompt length
        if prompt_length < 500:
            return ContextProfile.STANDARD
        elif prompt_length < 2000:
            return ContextProfile.EXTENDED
        else:
            return ContextProfile.MAXIMUM

    def optimize(
        self,
        prompt: str = "",
        messages: Optional[List[Dict[str, str]]] = None,
        system_prompt: str = "",
        expected_response_length: int = 512,
        priority: str = "balanced",
        max_memory_mb: Optional[float] = None
    ) -> ContextOptimizationResult:
        """
        Analyze input and generate optimal context configuration.

        Args:
            prompt: Current prompt text
            messages: Conversation history (list of {"role": str, "content": str})
            system_prompt: System prompt text
            expected_response_length: Expected response length in tokens
            priority: "speed", "balanced", or "quality"
            max_memory_mb: Override for available memory

        Returns:
            ContextOptimizationResult with recommended configuration
        """
        warnings = []
        suggestions = []

        # Get available memory
        available_mb = max_memory_mb or self.memory_analyzer.get_available_memory_mb()

        # Estimate token counts
        prompt_tokens = self.token_estimator.estimate(prompt).estimated_tokens
        system_tokens = self.token_estimator.estimate(system_prompt).estimated_tokens

        conversation_tokens = 0
        conversation_turns = 0
        if messages:
            conversation_tokens = self.token_estimator.estimate_conversation(messages)
            conversation_turns = len(messages)

        total_input_tokens = prompt_tokens + system_tokens + conversation_tokens

        # Detect content type
        has_code = any(indicator in (prompt + system_prompt)
                      for indicator in ["```", "def ", "function ", "class "])
        has_documents = len(prompt) > 3000 or "document" in prompt.lower()

        # Analyze use case
        recommended_profile = self.analyze_use_case(
            prompt_length=len(prompt),
            expected_response_length=expected_response_length,
            conversation_turns=conversation_turns,
            has_code=has_code,
            has_documents=has_documents,
            priority=priority
        )

        # Get base configuration
        config = self.get_profile_config(recommended_profile)

        # Adjust based on available memory
        max_ctx_for_memory = self.memory_analyzer.max_context_for_memory(
            available_mb, config.kv_cache_type
        )

        if config.num_ctx > max_ctx_for_memory:
            original_ctx = config.num_ctx
            config = ContextConfig(
                num_ctx=max_ctx_for_memory,
                num_batch=min(config.num_batch, max_ctx_for_memory // 4),
                num_predict=min(config.num_predict, max_ctx_for_memory // 2),
                system_prompt_budget=config.system_prompt_budget,
                kv_cache_type=self.memory_analyzer.recommend_kv_cache_type(
                    available_mb, max_ctx_for_memory
                ),
                rope_frequency_base=config.rope_frequency_base,
                rope_frequency_scale=config.rope_frequency_scale,
                sliding_window=min(config.sliding_window, max_ctx_for_memory),
                sliding_window_overlap=min(
                    config.sliding_window_overlap, max_ctx_for_memory // 4
                ),
            )
            warnings.append(
                f"Context reduced from {original_ctx} to {max_ctx_for_memory} due to memory constraints"
            )

        # Check if input exceeds context
        required_context = total_input_tokens + expected_response_length + 100  # buffer
        if required_context > config.num_ctx:
            if config.sliding_window > 0:
                suggestions.append(
                    f"Input ({total_input_tokens} tokens) exceeds context. "
                    f"Sliding window will be used to manage context."
                )
            else:
                warnings.append(
                    f"Input ({total_input_tokens} tokens) + expected response ({expected_response_length}) "
                    f"exceeds context limit ({config.num_ctx}). Consider summarizing or using extended profile."
                )

        # Calculate estimates
        kv_memory = self.memory_analyzer.calculate_kv_cache_memory(
            config.num_ctx, kv_type=config.kv_cache_type
        )

        # Rough latency estimate (varies significantly by hardware)
        # Base: 10ms per 100 tokens of context, plus batch processing overhead
        estimated_latency = (config.num_ctx / 100) * 10 + (config.num_batch / 100) * 5
        if config.kv_cache_type == "f16":
            estimated_latency *= 1.0
        elif config.kv_cache_type == "q8_0":
            estimated_latency *= 0.85
        else:
            estimated_latency *= 0.7

        # Rough tokens/second estimate
        tokens_per_second = 1000 / (estimated_latency / 10 + 50)

        # Add optimization suggestions
        if priority != "speed" and config.num_ctx > 2048:
            suggestions.append(
                f"Consider using 'quick_response' profile for simple queries to reduce latency"
            )

        if system_tokens > config.system_prompt_budget:
            suggestions.append(
                f"System prompt ({system_tokens} tokens) exceeds budget ({config.system_prompt_budget}). "
                f"Consider condensing for faster processing."
            )

        return ContextOptimizationResult(
            recommended_profile=recommended_profile,
            config=config,
            estimated_latency_ms=estimated_latency,
            memory_usage_mb=kv_memory,
            tokens_per_second_estimate=tokens_per_second,
            warnings=warnings,
            suggestions=suggestions,
        )

    def create_efficient_system_prompt(
        self,
        base_prompt: str,
        max_tokens: int = 256,
        preserve_instructions: bool = True
    ) -> str:
        """
        Create an efficient system prompt within token budget.

        Args:
            base_prompt: Original system prompt
            max_tokens: Maximum tokens for system prompt
            preserve_instructions: Try to keep key instructions

        Returns:
            Condensed system prompt
        """
        estimate = self.token_estimator.estimate(base_prompt)

        if estimate.estimated_tokens <= max_tokens:
            return base_prompt

        # Try to condense by removing redundant whitespace and verbose phrases
        condensed = base_prompt

        # Remove multiple spaces
        condensed = re.sub(r'\s+', ' ', condensed)

        # Remove common verbose phrases
        verbose_patterns = [
            (r'\bplease\b\s*', ''),
            (r'\bkindly\b\s*', ''),
            (r'\bmake sure to\b', ''),
            (r'\bremember to\b', ''),
            (r'\bdon\'t forget to\b', ''),
            (r'\bit is important that\b', ''),
            (r'\bthe following\b', 'these'),
            (r'\bin order to\b', 'to'),
            (r'\bfor the purpose of\b', 'for'),
        ]

        for pattern, replacement in verbose_patterns:
            condensed = re.sub(pattern, replacement, condensed, flags=re.IGNORECASE)

        # If still too long, truncate with ellipsis
        if self.token_estimator.estimate(condensed).estimated_tokens > max_tokens:
            # Estimate characters to keep
            chars_to_keep = int(max_tokens * self.token_estimator.chars_per_token * 0.9)
            if preserve_instructions:
                # Try to keep complete sentences
                sentences = condensed.split('. ')
                kept = []
                current_length = 0
                for sentence in sentences:
                    if current_length + len(sentence) + 2 <= chars_to_keep:
                        kept.append(sentence)
                        current_length += len(sentence) + 2
                    else:
                        break
                condensed = '. '.join(kept)
                if kept and not condensed.endswith('.'):
                    condensed += '.'
            else:
                condensed = condensed[:chars_to_keep].rsplit(' ', 1)[0] + '...'

        return condensed.strip()

    def get_ollama_options(
        self,
        config: ContextConfig,
        additional_options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate Ollama API options from configuration.

        Args:
            config: Context configuration
            additional_options: Additional options to merge

        Returns:
            Dictionary of Ollama options
        """
        options = {
            "num_ctx": config.num_ctx,
            "num_batch": config.num_batch,
        }

        if config.num_predict > 0:
            options["num_predict"] = config.num_predict

        # Add rope scaling for extended context
        if config.rope_frequency_base != 10000.0:
            options["rope_frequency_base"] = config.rope_frequency_base
        if config.rope_frequency_scale != 1.0:
            options["rope_frequency_scale"] = config.rope_frequency_scale

        if additional_options:
            options.update(additional_options)

        return options


class ContextProfileManager:
    """Manages context profiles from YAML configuration."""

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize with optional config path."""
        self.config_path = config_path
        self.profiles: Dict[str, Dict[str, Any]] = {}
        self._load_profiles()

    def _load_profiles(self):
        """Load profiles from YAML configuration."""
        if not self.config_path or not self.config_path.exists():
            return

        if not YAML_AVAILABLE:
            log.warning("PyYAML not available, using default profiles")
            return

        try:
            with open(self.config_path) as f:
                data = yaml.safe_load(f)
                if data and "profiles" in data:
                    self.profiles = data["profiles"]
                    log.info(f"Loaded {len(self.profiles)} context profiles")
        except Exception as e:
            log.warning(f"Error loading profiles: {e}")

    def get_profile(self, name: str) -> Optional[ContextConfig]:
        """Get a profile by name."""
        if name not in self.profiles:
            return None

        p = self.profiles[name]
        return ContextConfig(
            num_ctx=p.get("num_ctx", 2048),
            num_batch=p.get("num_batch", 256),
            num_predict=p.get("num_predict", 512),
            system_prompt_budget=p.get("system_prompt_budget", 256),
            kv_cache_type=p.get("kv_cache_type", "f16"),
            rope_frequency_base=p.get("rope_frequency_base", 10000.0),
            rope_frequency_scale=p.get("rope_frequency_scale", 1.0),
            sliding_window=p.get("sliding_window", 0),
            sliding_window_overlap=p.get("sliding_window_overlap", 0),
        )

    def list_profiles(self) -> List[str]:
        """List available profile names."""
        return list(self.profiles.keys())


def find_root() -> Path:
    """Locate USB-AI root directory."""
    script_dir = Path(__file__).parent.resolve()

    if (script_dir.parent.parent / "modules").exists():
        return script_dir.parent.parent
    if (script_dir.parent / "modules").exists():
        return script_dir.parent
    if (script_dir / "modules").exists():
        return script_dir

    # Fallback to working directory
    cwd = Path.cwd()
    if (cwd / "modules").exists():
        return cwd

    return script_dir.parent.parent


def main() -> int:
    """Command-line interface for context optimizer."""
    import argparse

    parser = argparse.ArgumentParser(
        description="USB-AI Context Window Optimizer"
    )
    parser.add_argument(
        "--profile",
        choices=["quick_response", "standard", "extended", "maximum"],
        default=None,
        help="Use specific profile (auto-detect if not specified)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="",
        help="Analyze optimization for this prompt"
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default="",
        help="System prompt to include in analysis"
    )
    parser.add_argument(
        "--priority",
        choices=["speed", "balanced", "quality"],
        default="balanced",
        help="Optimization priority"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llama",
        help="Model family for token estimation"
    )
    parser.add_argument(
        "--estimate-tokens",
        type=str,
        default=None,
        help="Estimate tokens for this text"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON"
    )
    parser.add_argument(
        "--create-config",
        action="store_true",
        help="Create context_profiles.yaml configuration file"
    )

    args = parser.parse_args()

    # Handle token estimation request
    if args.estimate_tokens:
        estimator = TokenEstimator(args.model)
        result = estimator.estimate(args.estimate_tokens)
        if args.json:
            print(json.dumps(asdict(result), indent=2))
        else:
            print(f"Text length: {result.text_length} characters")
            print(f"Estimated tokens: {result.estimated_tokens}")
            print(f"Chars per token: {result.chars_per_token:.2f}")
        return 0

    # Handle config creation
    if args.create_config:
        root = find_root()
        config_path = root / "modules" / "config" / "context_profiles.yaml"

        if config_path.exists():
            print(f"Configuration already exists at {config_path}")
            return 1

        # Create the config file
        config_content = create_default_config()
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w") as f:
            f.write(config_content)

        print(f"Created context profiles configuration at {config_path}")
        return 0

    print("")
    print("=" * 60)
    print("           USB-AI Context Window Optimizer")
    print("=" * 60)
    print("")

    # Initialize optimizer
    optimizer = ContextOptimizer(args.model)
    memory_analyzer = MemoryAnalyzer()

    # Show system info
    available_memory = memory_analyzer.get_available_memory_mb()
    print(f"Available Memory: {available_memory:.0f} MB")
    print(f"Model Family: {args.model}")
    print(f"Priority: {args.priority}")
    print("")

    # Run optimization
    if args.profile:
        profile = ContextProfile(args.profile)
        result = ContextOptimizationResult(
            recommended_profile=profile,
            config=optimizer.get_profile_config(profile),
            estimated_latency_ms=0,
            memory_usage_mb=0,
            tokens_per_second_estimate=0,
        )
    else:
        result = optimizer.optimize(
            prompt=args.prompt,
            system_prompt=args.system_prompt,
            priority=args.priority,
        )

    if args.json:
        output = {
            "profile": result.recommended_profile.value,
            "config": asdict(result.config),
            "estimated_latency_ms": result.estimated_latency_ms,
            "memory_usage_mb": result.memory_usage_mb,
            "tokens_per_second_estimate": result.tokens_per_second_estimate,
            "warnings": result.warnings,
            "suggestions": result.suggestions,
        }
        print(json.dumps(output, indent=2))
        return 0

    # Display results
    print(f"Recommended Profile: {result.recommended_profile.value}")
    print("")
    print("Configuration:")
    print(f"  Context Window:     {result.config.num_ctx} tokens")
    print(f"  Batch Size:         {result.config.num_batch}")
    print(f"  Max Predict:        {result.config.num_predict} tokens")
    print(f"  System Budget:      {result.config.system_prompt_budget} tokens")
    print(f"  KV Cache Type:      {result.config.kv_cache_type}")

    if result.config.sliding_window > 0:
        print(f"  Sliding Window:     {result.config.sliding_window} tokens")
        print(f"  Window Overlap:     {result.config.sliding_window_overlap} tokens")

    print("")
    print("Estimates:")
    print(f"  KV Cache Memory:    {result.memory_usage_mb:.1f} MB")
    print(f"  Est. Latency:       {result.estimated_latency_ms:.0f} ms")
    print(f"  Est. Tokens/sec:    {result.tokens_per_second_estimate:.1f}")

    if result.warnings:
        print("")
        print("Warnings:")
        for warning in result.warnings:
            print(f"  - {warning}")

    if result.suggestions:
        print("")
        print("Suggestions:")
        for suggestion in result.suggestions:
            print(f"  - {suggestion}")

    print("")
    print("Ollama Options:")
    options = optimizer.get_ollama_options(result.config)
    print(f"  {json.dumps(options)}")

    print("")

    return 0


def create_default_config() -> str:
    """Create default context_profiles.yaml content."""
    return '''# USB-AI Context Profiles Configuration
# ======================================
#
# This file defines context profiles for optimizing inference latency.
# Each profile is tuned for different use cases and response speed requirements.
#
# Profiles:
#   - quick_response: Minimal context for ultra-fast responses
#   - standard: Balanced speed and context for general use
#   - extended: Extended context for multi-turn conversations
#   - maximum: Maximum context for long documents and analysis
#
# Context Window Notes:
#   - Smaller context = faster first token latency
#   - Larger context = more conversation history retained
#   - KV cache type affects memory usage (f16 > q8_0 > q4_0)
#   - Sliding window helps manage very long conversations

version: "1.0.0"

# Default profile when not specified
default_profile: standard

profiles:
  quick_response:
    description: "Ultra-fast responses with minimal context (512 tokens)"
    num_ctx: 512
    num_batch: 128
    num_predict: 256
    system_prompt_budget: 128
    kv_cache_type: "q4_0"
    rope_frequency_base: 10000.0
    rope_frequency_scale: 1.0
    sliding_window: 0
    sliding_window_overlap: 0

    # Performance targets
    target_latency_ms: 50
    target_memory_mb: 64

    # Use cases
    use_cases:
      - "Quick Q&A"
      - "Simple commands"
      - "Status checks"
      - "One-word answers"

    # Recommended system prompt
    system_prompt_template: "Be concise. Answer in 1-2 sentences."

  standard:
    description: "Balanced speed and context for general use (2048 tokens)"
    num_ctx: 2048
    num_batch: 256
    num_predict: 512
    system_prompt_budget: 256
    kv_cache_type: "q8_0"
    rope_frequency_base: 10000.0
    rope_frequency_scale: 1.0
    sliding_window: 0
    sliding_window_overlap: 0

    target_latency_ms: 100
    target_memory_mb: 256

    use_cases:
      - "General conversation"
      - "Short coding tasks"
      - "Explanations"
      - "Writing assistance"

    system_prompt_template: |
      You are a helpful AI assistant. Provide clear and accurate responses.
      Be concise but thorough.

  extended:
    description: "Extended context for longer conversations (4096 tokens)"
    num_ctx: 4096
    num_batch: 512
    num_predict: 1024
    system_prompt_budget: 512
    kv_cache_type: "f16"
    rope_frequency_base: 10000.0
    rope_frequency_scale: 1.0
    sliding_window: 4096
    sliding_window_overlap: 1024

    target_latency_ms: 200
    target_memory_mb: 512

    use_cases:
      - "Multi-turn conversations"
      - "Code review"
      - "Document analysis"
      - "Complex problem solving"

    system_prompt_template: |
      You are a helpful AI assistant with expertise in multiple domains.
      Provide detailed and well-structured responses.
      Consider the full context of the conversation.

  maximum:
    description: "Maximum context for long documents (8192+ tokens)"
    num_ctx: 8192
    num_batch: 1024
    num_predict: 2048
    system_prompt_budget: 1024
    kv_cache_type: "f16"
    rope_frequency_base: 10000.0
    rope_frequency_scale: 1.0
    sliding_window: 8192
    sliding_window_overlap: 2048

    target_latency_ms: 500
    target_memory_mb: 1024

    use_cases:
      - "Long document analysis"
      - "Book summaries"
      - "Complex code refactoring"
      - "Research synthesis"

    system_prompt_template: |
      You are an expert AI assistant capable of analyzing complex documents
      and maintaining context across long conversations.
      Provide comprehensive, well-organized responses.
      Reference specific parts of the document when relevant.

# Memory-based profile selection
# Automatically select profile based on available memory
memory_auto_select:
  enabled: true
  thresholds:
    - max_memory_mb: 2048
      profile: quick_response
    - max_memory_mb: 4096
      profile: standard
    - max_memory_mb: 8192
      profile: extended
    - max_memory_mb: 999999
      profile: maximum

# Model-specific overrides
# Some models perform better with specific settings
model_overrides:
  dolphin-llama3:
    num_batch_multiplier: 1.0
    context_efficiency: 0.95

  qwen2.5:
    num_batch_multiplier: 1.2
    context_efficiency: 0.92

  mistral:
    num_batch_multiplier: 1.1
    context_efficiency: 0.94

  phi:
    num_batch_multiplier: 0.8
    context_efficiency: 0.90

# Conversation management
conversation:
  # Trigger context compression when this full
  compression_threshold: 0.85

  # Method for handling context overflow
  overflow_strategy: "sliding_window"  # or "summarize", "truncate"

  # Keep system prompt in context always
  preserve_system_prompt: true

  # Number of recent messages to always keep
  keep_recent_messages: 4

# Token estimation settings
token_estimation:
  # Default characters per token (varies by model)
  default_chars_per_token: 3.5

  # Adjustment for code content
  code_multiplier: 1.15

  # Adjustment for heavily punctuated text
  punct_threshold: 0.1
  punct_multiplier: 1.5

# Latency optimization settings
latency_optimization:
  # Prefill optimization
  enable_prefill_chunking: true
  prefill_chunk_size: 512

  # Speculative decoding (if supported)
  enable_speculative_decoding: false
  draft_model: null

  # KV cache optimization
  enable_kv_cache_quantization: true
  kv_cache_type_by_priority:
    speed: "q4_0"
    balanced: "q8_0"
    quality: "f16"
'''


if __name__ == "__main__":
    sys.exit(main())
