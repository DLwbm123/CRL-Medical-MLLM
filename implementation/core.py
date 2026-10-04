"""Independent implementation of arXiv:2511.17938v2 equations 2--21.

No author code or claimed author hyperparameters. All statistics exclude prompts
and padding. Vocabulary entropy and KL are exact sums, evaluated in FP32.
"""
import re
import unicodedata
from collections import Counter

import torch
from torch.utils.checkpoint import checkpoint

INPUT_KEYS = {"id", "source_id", "dataset", "split", "question", "options", "images", "image_paths", "language"}


def prompt_messages(row):
    if set(row) != INPUT_KEYS:
        raise ValueError("Adaptation input must match the label-free field allowlist")
    if len(row["images"]) != len(row["image_paths"]) or not row["images"]:
        raise ValueError("Invalid ordered image list")
    text = row["question"]
    if row["options"] is not None:
        text += "\nOptions:\n" + "\n".join(f"{key}. {value}" for key, value in row["options"].items())
        text += "\nReason through the question. End with a single line: Final answer: <option letter>."
    elif row["language"] == "zh":
        text += "\n请结合图像推理，最后用简短答案结束，格式为：最终答案：<答案>。"
    else:
        text += "\nReason through the question. End with a short answer on a single line: Final answer: <answer>."
    return [{"role": "user", "content": [{"type": "image", "image": p} for p in row["image_paths"]] + [{"type": "text", "text": text}]}]


def extract_answer(text, options=None):
    """Declared pilot parser, not the unavailable authors' grade_answer.

    Require an explicit answer marker (or a bare MC letter). Invalid completions
    never become a rewarded consensus. No reference labels are accepted.
    """
    text = unicodedata.normalize("NFKC", text).strip()
    matches = re.findall(r"(?:final[ \t]+answer|(?:the[ \t]+)?correct[ \t]+answer(?:[ \t]+is)?|answer|最终答案|答案)[ \t]*[:：][ \t]*([^\s\n][^\n]*)", text, flags=re.I)
    value = matches[-1].strip() if matches else text
    if options is not None:
        if matches:
            # Accept a leading option letter with optional option text. The
            # explicit letter wins; never infer an option from clinical prose.
            match = re.match(r"[\s*({\[]*([A-Za-z])(?=[\s*)}\].。,：:]|$)", value)
        else:
            match = re.fullmatch(r"[\s*({\[]*([A-Za-z])[\s*)}\].。]*", value)
        answer = match.group(1).upper() if match else None
        return answer if answer in options else None
    if not matches:
        return None
    value = re.sub(r"\s+", " ", value).casefold().strip(" *\t.。")
    return value or None


def consensus_rewards(completions, options=None):
    answers = [extract_answer(text, options) for text in completions]
    counts = Counter(answer for answer in answers if answer is not None)
    if not counts:
        raise ValueError("No parseable generated answers; cannot form a pseudo-reward")
    # Declared tie rule: lexicographic among most frequent valid answers.
    winner = min(counts, key=lambda answer: (-counts[answer], answer))
    rewards = torch.tensor([float(answer == winner) if answer is not None else 0.0 for answer in answers])
    advantages = (rewards - rewards.mean()) / (rewards.std(correction=0) + 1e-6)
    return rewards, advantages, {"answers": answers, "winner": winner, "valid": sum(a is not None for a in answers), "counts": dict(counts)}


def select_and_band(entropy, bins=100):
    values = entropy.detach().float()
    if values.ndim != 1 or values.numel() == 0 or not torch.isfinite(values).all():
        raise ValueError("Expected finite, nonempty response-token entropies")
    low, high = values.min(), values.max()
    threshold = high
    if high > low:
        mass = torch.histc(values, bins=bins, min=low.item(), max=high.item())
        mass = mass / mass.sum()
        centers = low + (torch.arange(bins, device=values.device) + 0.5) * (high - low) / bins
        w0 = mass.cumsum(0)[:-1]
        w1 = 1 - w0
        valid = (w0 > 0) & (w1 > 0)
        numerator = (mass * centers).cumsum(0)[:-1]
        total = (mass * centers).sum()
        score = torch.full_like(w0, -torch.inf)
        score[valid] = w0[valid] * w1[valid] * (numerator[valid] / w0[valid] - (total - numerator[valid]) / w1[valid]).square()
        if valid.any():
            threshold = centers[score.argmax()]  # First maximum if tied.
    selected = values >= threshold
    chosen = values[selected]
    center = chosen.quantile(0.5)  # Linear median interpolation, declared choice.
    mad = (chosen - center).abs().quantile(0.5)
    scale = (1.4826 * mad).clamp_min(1e-6)
    return selected, (center - scale).clamp_min(0), center, threshold


def _stats(logits, token_ids, reference_logits):
    logp = logits.float().log_softmax(-1)
    prob = logp.exp()
    chosen_logp = logp.gather(-1, token_ids[:, None]).squeeze(-1)
    entropy = -(prob * logp).sum(-1)
    if reference_logits.numel():
        ref_logp = reference_logits.float().log_softmax(-1)
        kl = (prob * (logp - ref_logp)).sum(-1)
    else:
        kl = torch.zeros_like(entropy)
    return chosen_logp, entropy, kl


def token_statistics(logits, token_ids, reference_logits=None, chunk_tokens=128):
    parts = []
    for start in range(0, len(token_ids), chunk_tokens):
        chunk = logits[start:start + chunk_tokens]
        ids = token_ids[start:start + chunk_tokens]
        reference = reference_logits[start:start + chunk_tokens].to(chunk.device) if reference_logits is not None else chunk.new_empty(0)
        if torch.is_grad_enabled() and chunk.requires_grad:
            parts.append(checkpoint(_stats, chunk, ids, reference, use_reentrant=False))
        else:
            parts.append(_stats(chunk, ids, reference))
    return tuple(torch.cat([part[i] for part in parts]) for i in range(3))


def objective(logp, old_logp, entropy, kl, advantage, selection, lower, upper, total_tokens, total_selected, cfg):
    ratio = (logp - old_logp).exp()
    surrogate = torch.minimum(ratio * advantage, ratio.clamp(1 - cfg["clip_epsilon"], 1 + cfg["clip_epsilon"]) * advantage)
    mask = selection.to(entropy.dtype)
    policy_loss = -(surrogate * mask).sum() / total_tokens
    band = ((lower - entropy).relu() * cfg["beta_lower"] + (entropy - upper).relu() * cfg["beta_upper"])
    band_loss = (band * mask).sum() / total_tokens if cfg["method"] == "SPINE" else entropy.sum() * 0
    kl_loss = cfg["kl_coefficient"] * (kl * mask).sum() / total_selected
    loss = policy_loss + band_loss + kl_loss
    return loss, {"policy": policy_loss.detach().item(), "band": band_loss.detach().item(), "kl": kl_loss.detach().item()}
