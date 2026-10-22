# Scientific semantics and explicit versions

## Preserved

CSRG uses two identical baseline forwards, four single gates and six pair gates.
For group baseline q10 `m`, single-gate q10 `m_i` and pair q10 `m_ij`:

```
necessity_i = max(0, m - m_i)
backup_ij = max(0, m - m_ij - necessity_i - necessity_j)
single_slack = min_i m_i
double_slack = min_ij m_ij
allocation_i = necessity_i / sum(necessity)        [when denominator > 0]
concentration = sum(allocation_i**2)
effective_support = 1 / concentration
```

Adam component load measurements preserve parameter RMS, first-moment RMS,
sqrt(mean second moment), and the frozen `m/(sqrt(v)+1e-8)` RMS. That last fixed
measurement epsilon is not a claim to be every optimizer's actual direction.

Forks use real post-minus-pre updates, not inferred gradients as substitutes.
Full, parameter-only and memory-only effects are measured against the matched
skip branch; the interaction residual is full minus parameter minus memory.
All 12 forward readouts and seven support arrays are retained (67 response roles).
Categorical changes are not coerced into continuous derivatives.

Finite-amplitude `_numeric_path` is AST-identical to the frozen function. Scale
grid, centers, epsilon, central differences and Simpson weights are unchanged.
The response is computed and sealed before native continuation.

Identity-ledger prediction uses `m + J + K/2`, margin sign, stable sample identity,
group capability means, and signs of capability differences. Prediction is
sealed before the alpha=1 endpoint. This is the identity-margin experiment's
candidate, not the distinct TL-P01 `quadratic_complete` algorithm.

## Versioned corrections / explicit extensions

1. **CSRG adapted v1.1**: undefined allocation/concentration are NaN with masks
   and ExplicitDisposition, not numerical zero. Defined-domain formulas match.
2. **Reciprocal classifier v1.1**: any uninformative required signature makes
   classification NOT_EVALUABLE. Fully Boolean rows retain the frozen rule.
3. **Parameter-path v1**: buffers held initially. Historical ResNet combined-state
   paths are not asserted equivalent. Native full forks/continuations retain real
   BN effects and record them separately.
4. **Explicit identity contracts**: stable sample IDs plus source/content hashes
   replace anonymous batch positions. This does not relabel old evidence.
5. **Undefined diagnostics**: no finite RMS observations -> None; zero or nonfinite
   ledger margins -> NOT_EVALUABLE under the default contract. The receiver-zero
   check is conservative and explicit; it does not alter historical labels.
6. **SGD Momentum**: separate native-state extension, not an Adam formula applied
   to SGD. Tested dampening/Nesterov cases are recorded in the validation report.

These are new output/protocol versions. No new claim has been retroactively
assigned to a frozen result, and no old file is silently repaired.

## What a PASS means

PASS means the implemented instrument executed under its declared contract,
restored/isolated state, retained actual outputs and passed numerical and GFG
checks. It does not establish a new scientific conclusion, historical model
performance, universal model portability, or adversarial proof that a dishonest
producer could not fabricate a journal. Graph integrity is not scientific truth.
