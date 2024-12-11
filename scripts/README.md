# scripts
This directory contains standalone python scripts that are useful to obtain some pieces of information or to manipulate data for certain purposes.

> These files must not be imported in any other files in the project, but only used in the terminal as standalone scripts

- `check_dataset.py`: DEPRECATED, remove
- `create_category_mapping`: Used to create the mapping between movies id and their relative category. TODO this may be inserted as a utility function and called in `get_category_mapping.py` if the category mapping still hasn't been created;
- `merge_run_sensitivity.py`: used to merge information about the model sensitivity with the probability to found a counterfactual.
- `round_csv.py`: rounds the float values of a csv to 3 decimal places.
