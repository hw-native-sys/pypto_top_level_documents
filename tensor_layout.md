# Tensor `tile_shape`: Tile-Contiguous Memory Layout for Efficient TLOAD/TSTORE

## Overview

This document specifies an optional `tile_shape` attribute for GM (Global Memory) tensors in pypto. By declaring a `tile_shape`, the programmer instructs the compiler/PTOAS to lay out the tensor in **tile-contiguous** order rather than the default **row-major** order. When the program later issues `TLOAD` / `TSTORE` operations using exactly that tile shape, every transfer becomes a single contiguous burst, maximizing GM bandwidth, NoC bandwidth, and L2 cache efficiency.

`tile_shape` is purely a **physical layout hint**. The logical `shape` of the tensor, the type system, and program semantics are unchanged.

## 1. Problem Statement

`TLOAD` and `TSTORE` instructions transfer a sub-tensor (a tile / slice) between GM and on-core SRAM (the local memory of an AIV or AIC core). The Ascend memory subsystem — the GM controller, the network-on-chip (NoC), and the cache hierarchy — is optimized around a target burst length and an L2 cache line size:

| Platform | GM/NoC burst length | L2 cache line |
|----------|--------------------|---------------|
| A2 / A3  | 512 B              | 512 B         |
| A5       | 128 B              | 512 B         |

Today, when the programmer defines a GM tensor by giving only its `shape`, the compiler defaults to a **row-major** layout: the bytes along the lowest (innermost) dimension are placed consecutively in memory, then the next lowest, and so on.

This default is fine when programs stream along the innermost dimension. It is **catastrophic** when a kernel accesses small 2-D (or higher-rank) tiles that span multiple rows of a wide tensor.

### Concrete Example

Consider a GM tensor of shape `(1024, 1024)` in FP8 (1 byte/element), laid out row-major:

```
row 0:  [ b0,0  b0,1  b0,2 ... b0,1023 ]   (1024 bytes contiguous)
row 1:  [ b1,0  b1,1  b1,2 ... b1,1023 ]   (1024 bytes contiguous)
...
row 1023: ...
```

Suppose a kernel issues a `TLOAD` for a `(16, 16)` FP8 tile starting at `(r0, c0)`. With the row-major layout, the 16 bytes of each tile-row are contiguous, but consecutive tile-rows are **1024 bytes apart**:

```
TLOAD (16, 16) tile from row-major (1024, 1024) FP8 tensor:

  load 0:  16 bytes at  base + (r0+0)*1024 + c0    (BL = 16 B)
  load 1:  16 bytes at  base + (r0+1)*1024 + c0    (BL = 16 B, stride = 1024 B)
  load 2:  16 bytes at  base + (r0+2)*1024 + c0    (BL = 16 B, stride = 1024 B)
  ...
  load 15: 16 bytes at  base + (r0+15)*1024 + c0   (BL = 16 B, stride = 1024 B)

  → 16 short bursts of 16 B each, with a large stride between them.
```

Consequences:

1. **Burst length = 16 B**, far below the optimal 512 B (A2/A3) or 128 B (A5). The GM controller and NoC are forced to issue many short transactions, paying per-transaction overhead on every one.
2. **L2 cache pollution.** Each 16 B fetch pulls in a full 512 B cache line, of which only 16 B is used; the remainder is wasted unless re-touched soon.
3. **NoC channel underutilization.** The wide pipes are starved by the small request size.
4. **Longer end-to-end runtime** of `TLOAD` / `TSTORE`, which often sit on the critical path of compute kernels.

## 2. Solution: Tile-Contiguous Layout via `tile_shape`

We extend the GM tensor definition (typically the output of a `submit` task, or any user-declared GM tensor) with an **optional** `tile_shape` argument:

```python
T = pl.gm_tensor(
        shape      = (M, N),       # logical shape (unchanged)
        dtype      = pl.fp8,
        tile_shape = (TM, TN),     # NEW: optional physical tile shape
)
```

### Rules

1. `tile_shape` is **optional**. If omitted, it defaults to `shape`, which reproduces today's row-major behavior — fully backward compatible.
2. `tile_shape` must have the **same number of dimensions** as `shape`.
3. Each dimension of `shape` must be a **multiple** of the corresponding dimension of `tile_shape`:

   ```
   ∀ i:  shape[i] % tile_shape[i] == 0
   ```

   The compiler shall reject definitions that violate this constraint.

### Memory Layout Definition

The tensor is partitioned into a regular grid of tiles of shape `tile_shape`. Tiles are enumerated in row-major order over the *tile grid* (lowest tile-grid dimension varies fastest). Within each tile, bytes are laid out contiguously in row-major order over the *tile interior*. The full physical layout is the concatenation of all tiles in tile-grid order:

```
physical layout =  [ tile(0,0) bytes ][ tile(0,1) bytes ] ... [ tile(0, N/TN - 1) bytes ]
                   [ tile(1,0) bytes ][ tile(1,1) bytes ] ... [ tile(1, N/TN - 1) bytes ]
                   ...
                   [ tile(M/TM - 1, 0) bytes ] ... [ tile(M/TM - 1, N/TN - 1) bytes ]
```

For the running example (`shape = (1024, 1024)`, FP8, `tile_shape = (16, 16)`), each tile is `16*16 = 256` consecutive bytes in GM:

```
GM byte offsets for tile_shape = (16, 16):

  offset      0 .. 255    : tile (0, 0)         (16x16 contiguous)
  offset    256 .. 511    : tile (0, 1)
  offset    512 .. 767    : tile (0, 2)
  ...
  offset 16128 .. 16383   : tile (0, 63)
  offset 16384 .. 16639   : tile (1, 0)
  ...
```

A `TLOAD` of a `(16, 16)` tile at tile-grid coordinate `(ti, tj)` is now a **single contiguous 256-byte burst** at offset `(ti * (N/TN) + tj) * 256`, instead of 16 strided 16-byte bursts.

### Visual Comparison

```
Default row-major layout, 16x16 TLOAD:

   row 0  ┌─[##]─────────────────────────────────────────┐
   row 1  ├─[##]─────────────────────────────────────────┤
   row 2  ├─[##]─────────────────────────────────────────┤   16 short bursts
   ...    │   ...                                          │   stride = 1024 B
   row 15 ├─[##]─────────────────────────────────────────┤   BL     = 16 B
          └────────────────────────────────────────────────┘

Tile-contiguous layout (tile_shape = (16,16)), same 16x16 TLOAD:

          ┌────────────────────────────────────────────────┐
          │ [################################]              │   1 burst
          │  ^ 256 contiguous bytes for the tile             │   BL = 256 B
          └────────────────────────────────────────────────┘
```

## 3. When and Why This Works

The key invariant is:

> If, during the lifetime of the GM tensor, the program **only** accesses it via `TLOAD` / `TSTORE` whose access shape equals the declared `tile_shape` (and is aligned to the tile grid), then **every** transfer becomes a contiguous burst whose length equals `prod(tile_shape) * sizeof(dtype)`.

When this invariant holds, tile-contiguous layout delivers:

- **Maximal GM burst length** — the burst length naturally equals the tile byte size, and the programmer chooses `tile_shape` so this matches (or is a multiple of) the platform's preferred burst length (512 B on A2/A3, 128 B on A5).
- **Maximal NoC efficiency** — wide channels carry full-size transactions instead of being starved by short, strided requests.
- **L2 cache line utilization → 100 %** — every fetched cache line is fully consumed by the tile.
- **Fewer outstanding transactions** — reduces controller queue pressure and tail latency.

### Caveats: When *not* to specify `tile_shape`

The optimization is only beneficial when the access pattern is congruent with the declared `tile_shape`. If the program also accesses the tensor with a *different* tile shape — for example a transposed or reshaped view, a strip along a single row, or a sub-tile smaller than `tile_shape` — the access reverts to short, strided bursts (often *worse* than the row-major default, because the addressing also becomes non-trivial).

Therefore:

- `tile_shape` shall only be specified when the dominant lifetime access pattern of the GM tensor matches it.
- The programmer must specify it **with discretion**, treating it as a performance contract.
- When in doubt, omit it — the compiler will fall back to row-major, and correctness is unaffected.

The compiler is **not** required to dynamically rewrite layouts based on observed access patterns. `tile_shape` is a programmer-driven hint, not an autotuner.

## 4. Impact on the pypto Compiler, PTOAS, and Simpler Runtime

### 4.1 pypto Compiler

#### Tensor metadata

The pypto IR's GM tensor structure must carry an additional optional field:

| Field        | Type             | Meaning                                               |
|--------------|------------------|-------------------------------------------------------|
| `shape`      | tuple of int/Expr | Logical shape (unchanged).                            |
| `dtype`      | type             | Element type (unchanged).                             |
| `tile_shape` | tuple of int/Expr or `None` | Physical tile shape; `None` ⇒ row-major default. |

The field must be **preserved** through every IR pass and **propagated** to the back end so that PTOAS can see it.

#### Validation

When a tensor is declared with `tile_shape`, the compiler must verify:

1. `len(tile_shape) == len(shape)`.
2. `shape[i] % tile_shape[i] == 0` for every dimension `i`.
3. `prod(tile_shape) * sizeof(dtype)` is a sensible burst size (not below platform minimum, not above tile SRAM budget). This may be a soft check producing a warning rather than an error.

Violations of (1) or (2) are hard errors.

#### Layout-affecting operations: warn the programmer

Operations that re-interpret or reshape a tensor's logical structure can silently break the tile-contiguity assumption. When a tensor with a non-default `tile_shape` is the input to any of the following, the compiler shall emit a **warning**:

- `tensor.slice` / `pl.slice` — when the slice does not align to the tile grid, or when the slice extents are not multiples of `tile_shape`, the slice cannot inherit `tile_shape` and falls back to a sub-region with non-trivial stride.
- `tensor.shallow_reshape` — a metadata-only reshape that assumes a contiguous row-major layout. With a non-default `tile_shape`, shallow reshape is generally **invalid** and must be rejected (or downgraded to a warning + propagated `tile_shape = None`).
- `tensor.deep_reshape` — a copying reshape. The compiler should warn that the destination tensor's layout will be the default row-major unless the user explicitly supplies a new `tile_shape` for the destination.
- Any other tensor layout / view operation (`transpose`, `permute`, `view`, `expand`, ...).

Default behavior on warning: produce the result tensor with `tile_shape = None` (row-major) so that subsequent codegen is correct, even if no longer optimal. The warning message must clearly identify the source location and suggest either dropping `tile_shape`, aligning the slice, or rematerializing with a fresh `tile_shape`.

#### Codegen

When generating `TLOAD` / `TSTORE` for a tensor with non-default `tile_shape`, the compiler computes the GM offset of the requested tile from the **tile-grid coordinates**, not the default row-major offset formula. This information is forwarded to PTOAS via the tensor's metadata.

### 4.2 PTOAS

PTOAS is responsible for emitting the physical instruction binary for `TLOAD`, `TSTORE`, and `TASSEMBLE`. Its requirements:

1. **Accept and honor the `tile_shape` metadata** for every GM tensor argument.
2. **Be cognizant of the layout** when emitting addressing logic. For a non-default `tile_shape`, the byte address of element `(i0, i1, ...)` is:

   ```
   tile_idx = (i0 / tile_shape[0], i1 / tile_shape[1], ...)
   intra   = (i0 % tile_shape[0], i1 % tile_shape[1], ...)

   addr    = base
           + linearize_row_major(tile_idx, shape / tile_shape) * prod(tile_shape) * sizeof(dtype)
           + linearize_row_major(intra,    tile_shape)         * sizeof(dtype)
   ```

   For row-major (`tile_shape == shape`), this collapses to today's address formula.

3. **Generate physical instructions whose access pattern matches the layout.** When the `TLOAD` / `TSTORE` access shape equals `tile_shape` and is tile-aligned, PTOAS shall emit a single contiguous-burst instruction. When it does not match, PTOAS may emit either a strided sequence (correct but slow) or refuse and report an error, per the platform-specific code generator's policy.
4. **`TASSEMBLE`** (which composes a tensor in SRAM from multiple sub-tile transfers) must use the same layout-aware addressing so that the in-SRAM image is reconstructed correctly regardless of GM layout.

PTOAS is the lowest level that has to *act* on `tile_shape`. The pypto compiler decides the layout; PTOAS executes it.

### 4.3 Simpler Runtime

The simpler distributed runtime is responsible for tensor lifetime tracking, dependency analysis (the tensormap), and argument marshalling. With respect to `tile_shape` it has minimal duties:

- **Tensormap / overlap analysis.** Continue to use only the logical `shape` (and offset/stride) to detect overlap and dependency between tensor regions. `tile_shape` does **not** alter logical extents and therefore does not affect overlap reasoning.
- **Data structure.** Carry `tile_shape` as an opaque field on the tensor descriptor and propagate it through the argument list to PTOAS-generated kernels and to debug/trace outputs.
- **Debug output.** When dumping a tensor descriptor (logs, traces, error messages), include `tile_shape` so that layout-related performance issues are inspectable.

At the time of this writing there is no other runtime use of `tile_shape`. Should a future pass need it (e.g., layout-aware scheduling, peer-to-peer transfer chunking), the field is already plumbed through.

## 5. pypto Grammar Enhancement: Implicit Dimension Padding for `pl.slice` and `pl.index`

This section introduces two grammar conveniences for accessing an n-D tensor or sub-tensor. Both let the programmer supply fewer coordinates than the rank of the tensor; the compiler then pads the missing dimensions according to a fixed, predictable rule. This removes a large amount of boilerplate `0` / full-extent arguments from kernels that index or slice high-rank tensors.

Throughout this section, dimensions are numbered from the outermost (dimension 0, varies slowest in row-major) to the innermost (dimension N-1, varies fastest). "Higher" dimension means closer to dimension 0 (outer); "lower" dimension means closer to dimension N-1 (inner).

**Signature note.** The canonical pypto signature is `pl.slice(tensor, shape, offset)` — the slice shape comes first, the offset second — consistent with all existing usages in the codebase. The expansions below are written using this ordering.

### 5.1 `pl.slice` — implicit lower-dimension expansion

When `pl.slice` is given an offset list whose length `k` is less than the tensor rank `N`, the abbreviated form expands as follows:

- **Offset.** The `k` provided values are taken as the offsets of the `k` highest (outermost) dimensions. The remaining `N − k` lower (inner) dimensions are padded with offset `0`.
- **Shape.** The `k` highest dimensions get extent `1`. The remaining `N − k` lower dimensions adopt the full extent of the corresponding dimension of the source tensor `A`.

Concretely, let `A` be a 4-D tensor with shape `(s0, s1, s2, s3)`. Then:

```python
B = pl.slice(A, [i, j])
```

is exactly equivalent to the fully-specified form:

```python
B = pl.slice(A, [1, 1, s2, s3], [i, j, 0, 0])
#                ^^^^^^^^^^^^^^  ^^^^^^^^^^^^
#                shape           offset
```

That is:

- the two higher dimensions are pinned to offsets `i` and `j` with extent `1` each (padding the lower-dimension offsets to `0`), and
- the two lower dimensions take the full source extents `s2` and `s3` (padding the higher-dimension shape entries with `1`).

Intuitively, `pl.slice(A, [i, j])` selects the full `(s2, s3)` sub-tensor located at the `(i, j)` position of the two outer dimensions — exactly what is wanted when the outer dimensions index a batch / block grid and the inner dimensions form the tile of interest.

### 5.2 `pl.index` — implicit higher-dimension padding

When `pl.index` is given an index list whose length `k` is less than the tensor rank `N`, the `k` provided values are taken as the indices of the `k` lowest (innermost) dimensions, and the remaining `N − k` higher (outermost) dimensions are padded with index `0`.

For the same 4-D tensor `A`:

```python
alpha = pl.index(A, [i, j])
```

is exactly equivalent to:

```python
alpha = pl.index(A, [0, 0, i, j])
```

i.e. the missing higher-dimension indices are filled with `0`, and the supplied `[i, j]` address the two innermost dimensions.

### 5.3 Summary of the padding rules

| Operation | Provided `k < N` values address … | Missing dimensions padded with … |
|-----------|-----------------------------------|----------------------------------|
| `pl.slice` offset | the `k` highest dimensions | lower-dim offsets → `0`; their shape → full source extent |
| `pl.slice` shape | extent `1` for those `k` dims | higher-dim shape entries → `1` |
| `pl.index` | the `k` lowest dimensions | higher-dim indices → `0` |

Note the deliberate asymmetry: `pl.slice` binds the abbreviated coordinates to the outer dimensions (taking a full inner sub-tensor), whereas `pl.index` binds them to the inner dimensions (addressing the innermost elements). Both choices match the most common access patterns for their respective operations.

These are purely front-end grammar conveniences. After expansion the operations have identical semantics to their fully-specified forms; the type system, IR, and back-end code generation are unchanged.

## Summary

| Component        | Action required |
|------------------|-----------------|
| Programmer       | Optionally declare `tile_shape` on a GM tensor when the dominant `TLOAD`/`TSTORE` access pattern is known. Use abbreviated `pl.slice(A, […])` to pin outer dimensions and take full inner extents, and abbreviated `pl.index(A, […])` to address inner dimensions with outer indices defaulting to `0`. |
| pypto compiler   | Carry `tile_shape` in the tensor IR; validate divisibility; warn on layout-breaking ops; forward to PTOAS. Expand abbreviated `pl.slice` / `pl.index` to their fully-specified forms per §5 before lowering; after expansion, semantics, IR, and back-end codegen are unchanged. |
| PTOAS            | Use `tile_shape` to compute GM addresses; emit single-burst instructions when access matches the tile. No direct handling of §5 grammar sugar — it is resolved upstream. |
| Simpler runtime  | Propagate `tile_shape` through descriptors and argument lists; ignore it for overlap/tensormap reasoning; surface it in debug output. No change for §5 grammar sugar. |

When used correctly, `tile_shape` turns small-tile `TLOAD` / `TSTORE` traffic from many short, strided bursts into one long contiguous burst, restoring full GM bandwidth, NoC throughput, and L2 cache utilization — directly shortening kernel runtime on A2, A3, and A5.

Abbreviated `pl.slice` and `pl.index` remove boilerplate `0` and full-extent arguments from high-rank indexing and slicing; the compiler expands them deterministically, so correctness and performance behavior match the fully-specified forms.
