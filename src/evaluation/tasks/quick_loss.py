"""Quick loss eval — fast proxy metric via batched forward pass.

No generation, no per-pair scoring, no downloads. Just runs the model
over held-out token sequences and measures:
  - Loss (cross-entropy)
  - Top-1 accuracy (next-token prediction)
  - Top-5 accuracy
  - Entropy of predictions (low = confident, high = confused)

Designed to run in seconds, not minutes. Correlates with heavier evals
because better next-token prediction → better everything else.
"""

from __future__ import annotations

import math
import time
from typing import Optional

import torch
import torch.nn.functional as F

from src.evaluation.config import EvalConfig
from src.evaluation.protocol import Evaluatable
from src.evaluation.results import EvalResult
from src.evaluation.tasks import register_task
from src.evaluation.tasks.base import EvalTask, ProgressCallback


# Short held-out sequences from common English patterns.
# These are NOT in TinyStories but use similar vocabulary/structure.
# Each is a self-contained snippet the model should assign reasonable probability to.
_HELD_OUT_TEXTS = [
    "Once upon a time there was a little girl who lived in a small house near the forest.",
    "The boy ran to the park and played with his friends until the sun went down.",
    "She opened the door and saw a big red ball sitting on the table.",
    "The old man smiled and said thank you to the kind woman.",
    "It was a cold winter morning and the snow covered everything in white.",
    "The dog barked loudly and ran across the yard to greet its owner.",
    "He picked up the book and began to read the first page carefully.",
    "The little bird sat on the branch and sang a beautiful song.",
    "They walked along the river and watched the fish swimming below.",
    "The mother called her children inside because it was time for dinner.",
    "A long time ago in a land far away there lived a wise old king.",
    "The cat jumped onto the bed and curled up into a warm little ball.",
    "She looked out the window and saw the rain falling from the dark clouds.",
    "The farmer planted seeds in the garden and watered them every morning.",
    "He was very happy because his best friend came to visit him today.",
    "The teacher asked the students to open their books to page ten.",
    "There was a bright star shining in the sky above the mountains.",
    "The baby laughed and clapped her hands when she saw the funny toy.",
    "They built a sandcastle on the beach and decorated it with shells.",
    "The princess wore a beautiful dress and danced at the grand ball.",
    "He tried to climb the tall tree but the branches were too high.",
    "The rabbit hopped through the meadow looking for something to eat.",
    "She put on her red coat and went outside to play in the snow.",
    "The wind blew hard and the leaves fell from the trees all around.",
    "Once there was a fox who was very clever and liked to play tricks.",
    "The fish swam in circles around the pond waiting to be fed.",
    "He found a shiny coin on the ground and put it in his pocket.",
    "The flowers bloomed in the spring and filled the air with a sweet smell.",
    "She was scared of the dark but her father held her hand tight.",
    "The train moved slowly along the tracks through the green valley.",
    "They shared the cake equally so that everyone got a piece.",
    "The moon rose above the hills and lit up the night sky.",
    # Common knowledge / basic facts
    "The sun rises in the east and sets in the west every single day.",
    "Water freezes when the temperature drops below zero degrees.",
    "Birds have wings and most of them can fly high up in the sky.",
    "Fish live in the water and breathe through their gills.",
    "The earth goes around the sun and the moon goes around the earth.",
    "Trees need water and sunlight to grow tall and strong.",
    "Fire is very hot and can burn you if you get too close.",
    "Ice cream melts when you leave it out in the warm sun.",
    "Cats can see very well in the dark because of their special eyes.",
    "Dogs are very loyal animals and they love to play with people.",
    # Cause and effect
    "Because it rained all day the streets were wet and full of puddles.",
    "She forgot her umbrella so she got very wet walking home from school.",
    "He ate too much candy and his stomach started to hurt.",
    "The plant died because nobody remembered to water it for two weeks.",
    "They were late to the party because the car broke down on the way.",
    "The ice melted quickly because the room was very warm inside.",
    "She studied very hard for the test and got the highest score in class.",
    "The road was icy so the cars had to drive very slowly and carefully.",
    # Dialogue and social patterns
    "Hello said the girl my name is Lily and I am five years old.",
    "Please can I have some more asked the hungry little boy at the table.",
    "Thank you so much said the woman you are very kind to help me.",
    "I am sorry said the boy I did not mean to break your favorite cup.",
    "Come here called the mother it is time to go home now.",
    "Look at that said the father pointing at the big rainbow in the sky.",
    # Sequences and counting
    "First she washed her hands then she sat down and ate her lunch.",
    "There were three little kittens and each one had a different color.",
    "He counted one two three four five and then he opened his eyes.",
    "Monday Tuesday Wednesday Thursday Friday are the days we go to school.",
    "In the morning she eats breakfast then she brushes her teeth and gets dressed.",
    # Emotions and descriptions
    "The little boy was so excited that he could not stop jumping up and down.",
    "She felt sad when her best friend moved away to another town.",
    "The scary noise in the dark made the children hide under the blanket.",
    "He was proud of himself because he finally learned how to ride a bike.",
    "The beautiful sunset painted the sky with colors of orange pink and gold.",
    "Everyone was surprised when the quiet little mouse scared the big cat away.",
    # Nature and animals
    "The butterfly flew from flower to flower collecting sweet nectar in the garden.",
    "Squirrels gather nuts in the fall and save them for the cold winter months.",
    "The frog jumped into the pond with a big splash and swam away.",
    "Bees make honey by collecting pollen from many different kinds of flowers.",
    "The bear went to sleep in his cave and did not wake up until spring.",
    "Baby ducks follow their mother in a line wherever she goes.",
    # Simple narrative arcs
    "The boy lost his favorite toy but after looking everywhere he finally found it under the bed.",
    "She was afraid to swim at first but her father taught her and soon she loved it.",
    "The two friends had a fight but the next day they said sorry and played together again.",
    "He wanted to bake a cake so he mixed the flour eggs and sugar in a big bowl.",
    "The little seed was planted in the ground and after many days it grew into a tall sunflower.",
    "They got lost in the woods but followed the sound of the river and found their way home.",
]


@register_task("quick_loss")
class QuickLossTask(EvalTask):
    """Fast eval via batched forward pass on held-out text snippets."""

    name = "quick_loss"

    def download(self, data_dir: str) -> None:
        pass  # No downloads needed

    def evaluate(
        self,
        model: Evaluatable,
        config: EvalConfig,
        on_progress: Optional[ProgressCallback] = None,
    ) -> EvalResult:
        tokenizer = model.tokenizer
        device = model.device

        max_texts = config.max_samples or len(_HELD_OUT_TEXTS)
        texts = _HELD_OUT_TEXTS[:max_texts]

        t0 = time.perf_counter()

        total_loss = 0.0
        total_top1_correct = 0
        total_top5_correct = 0
        total_tokens = 0
        total_entropy = 0.0

        for i, text in enumerate(texts):
            token_ids = tokenizer.encode(text)
            if len(token_ids) < 2:
                continue

            # Single forward pass per snippet
            ll = model.loglikelihood_rolling(token_ids)
            n_scored = len(token_ids) - 1

            # Loss from log-likelihood
            avg_nll = -ll / n_scored
            total_loss += avg_nll * n_scored

            # For top-k accuracy we need logits — use loglikelihood approach
            # but also check top-k. We do this efficiently via a single forward pass.
            ids_tensor = torch.tensor([token_ids], device=device)

            # We need raw logits for top-k, so do a direct forward pass
            with torch.inference_mode():
                # Access the underlying model if wrapped
                raw_model = getattr(model, '_model', None)
                if raw_model is not None:
                    was_training = raw_model.training
                    raw_model.eval()
                    logits = raw_model(ids_tensor).logits
                    if was_training:
                        raw_model.train()
                else:
                    # Fallback: just use log-likelihood scores
                    total_tokens += n_scored
                    continue

                # logits: (1, seq_len, vocab_size)
                # Predict token[t+1] from position t
                pred_logits = logits[0, :-1, :]  # (seq_len-1, vocab_size)
                targets = torch.tensor(token_ids[1:], device=device)

                # Top-1
                top1_preds = pred_logits.argmax(dim=-1)
                total_top1_correct += (top1_preds == targets).sum().item()

                # Top-5
                top5_preds = pred_logits.topk(min(5, pred_logits.size(-1)), dim=-1).indices
                total_top5_correct += (top5_preds == targets.unsqueeze(-1)).any(dim=-1).sum().item()

                # Entropy of predictions (measures confidence)
                probs = F.softmax(pred_logits, dim=-1)
                entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1)
                total_entropy += entropy.sum().item()

                total_tokens += n_scored

            if on_progress:
                on_progress(i + 1, len(texts))

        elapsed = time.perf_counter() - t0

        if total_tokens == 0:
            return EvalResult(
                task_name=self.name,
                metrics={"error": 1.0},
                metadata={"error": "No tokens scored"},
            )

        avg_loss = total_loss / total_tokens
        top1_acc = total_top1_correct / total_tokens
        top5_acc = total_top5_correct / total_tokens
        avg_entropy = total_entropy / total_tokens
        ppl = math.exp(min(avg_loss, 20))

        return EvalResult(
            task_name=self.name,
            metrics={
                "loss": round(avg_loss, 4),
                "perplexity": round(ppl, 2),
                "top1_accuracy": round(top1_acc, 4),
                "top5_accuracy": round(top5_acc, 4),
                "entropy": round(avg_entropy, 4),
            },
            metadata={
                "num_texts": len(texts),
                "total_tokens": total_tokens,
                "elapsed_seconds": round(elapsed, 2),
            },
        )
