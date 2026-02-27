# Region Sampling Notes

This note summarizes the current behavior of region sampling in `MultiRegionRandomSampler` and records concrete examples found during debugging.

## 1. What is sampled

Sampling is done on the object placement center `(object_x, object_y, object_z)`, not directly on object boundaries.

Core logic is in `base_region_sampler.py`:

- `_sample_x()` / `_sample_y()`
- `sample()` where `(object_x, object_y, object_z)` is written into `pos`

## 1.1 Where to check object radius (`r`)

For most XML-based objects used in this repo (e.g. HOPE / scanned objects), radius is defined by the XML site:

- `<site name="horizontal_radius_site" pos="...">`

Example:

- `libero/libero/assets/stable_hope_objects/alphabet_soup/alphabet_soup.xml`

Runtime usage in LIBERO samplers:

- `horizontal_radius = obj.horizontal_radius` in `base_region_sampler.py`

## 1.2 Robosuite default radius parsing rule

In robosuite's `MujocoXMLObject`, the default implementation is:

- find site `"{naming_prefix}horizontal_radius_site"`
- parse `pos`
- use only the first component (`x`) as scalar radius

Equivalent rule:

- `r = horizontal_radius_site.pos[0]`

Reference implementation (local robosuite source / site-packages):

- `robosuite/models/objects/objects.py`, property `horizontal_radius`

Note:

- some primitive / composite object classes can override `horizontal_radius`; this rule is the default for XML object models.

## 2. Effect of `ensure_object_boundary_in_range`

Given BDDL range `[xmin, xmax]` and object horizontal radius `r`:

- If `ensure_object_boundary_in_range=False`: sampled center range is `[xmin, xmax]`
- If `ensure_object_boundary_in_range=True`: sampled center range becomes `[xmin + r, xmax - r]`

Same rule for `y`.

So the width transforms as:

- original width: `w = xmax - xmin`
- effective width: `w_eff = w - 2r`

## 3. Three regimes when `ensure_object_boundary_in_range=True`

- `w > 2r`: normal (non-degenerate) interval
- `w = 2r`: degenerate to a single point
- `w < 2r`: interval flips (`min' > max'`)

Important: with `numpy.random.uniform(high, low)`, even if `high < low`, sampling still returns values between them (equivalent to sampling from `[high, low]`). This is why flipped intervals may still produce valid numeric samples, but semantics are no longer the intended "boundary-safe shrink".

## 4. Concrete examples from this repo

### 4.1 `libero_object_single` custom anchor-style BDDL

File:

`libero/libero/bddl_files/libero_object_single/pick_up_the_alphabet_soup_and_place_it_in_the_basket.bddl`

For one anchor patch:

- range: `(-0.36, -0.36, -0.34, -0.34)` (width `0.02`)
- `alphabet_soup` has `horizontal_radius_site` with x = `0.025` in XML
- using `r = 0.025`, we get `w < 2r` (`0.02 < 0.05`) -> interval flips

Result with `ensure_object_boundary_in_range=True`:

- raw x range: `[-0.335, -0.365]` (flipped)
- equivalent sampled x range: `[-0.365, -0.335]`
- same for y

Result with `ensure_object_boundary_in_range=False`:

- x in `[-0.36, -0.34]`
- y in `[-0.36, -0.34]`

This matches "sample around center anchor patch" intent.

### 4.2 Original repo BDDL (`libero_object`)

File:

`libero/libero/bddl_files/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket.bddl`

Observed outcomes with current defaults (`TableRegionSampler`, `ensure_object_boundary_in_range=True`):

- some objects collapse to points (`w = 2r`)
- some objects flip (`w < 2r`)
- some stay normal (`w > 2r`)

Examples:

- `alphabet_soup_1`: point `(-0.12, -0.24)`
- `salad_dressing_1`: point `(0.05, -0.10)`
- `tomato_sauce_1`: point `(0.15, 0.03)`
- `cream_cheese_1`: small interval after flip-equivalence
- `milk_1`: small interval after flip-equivalence

## 5. Practical guidance

If your BDDL regions are intended as anchor patches for center sampling (especially small discrete patches), prefer:

- `ensure_object_boundary_in_range=False`

If strict boundary-safe placement is desired, ensure each patch width satisfies:

- `w >= 2r` for all sampled objects

otherwise you will get point collapse or flipped intervals.
