import random
from typing import Tuple, List

def apply_fim(
    token_ids: List[int],
    fim_prefix_id: int,
    fim_middle_id: int,
    fim_suffix_id: int,
    fim_rate: float = 0.5,
    spm_rate: float = 0.5
) -> List[int]:
    """
    Aplica transformação Fill-In-the-Middle (FIM) em uma sequência de tokens.
    Suporta os dois modos canônicos de treino (Bavarian et al., OpenAI 2022):
    - PSM: <|fim_prefix|> Prefix <|fim_suffix|> Suffix <|fim_middle|> Middle
    - SPM: <|fim_suffix|> Suffix <|fim_prefix|> Prefix <|fim_middle|> Middle
    """
    if len(token_ids) < 8 or random.random() > fim_rate:
        return token_ids

    # Sorteia 2 pontos de corte dentro da sequência
    boundaries = sorted([random.randint(0, len(token_ids)) for _ in range(2)])
    b1, b2 = boundaries[0], boundaries[1]

    prefix = token_ids[:b1]
    middle = token_ids[b1:b2]
    suffix = token_ids[b2:]

    if random.random() < spm_rate:
        # Modo SPM
        return (
            [fim_suffix_id] + suffix +
            [fim_prefix_id] + prefix +
            [fim_middle_id] + middle
        )
    else:
        # Modo PSM
        return (
            [fim_prefix_id] + prefix +
            [fim_suffix_id] + suffix +
            [fim_middle_id] + middle
        )
