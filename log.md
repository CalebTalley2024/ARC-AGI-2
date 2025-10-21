## 10-7-25 @caleb 

- part 2: Tasks complete, *requirements.txt did not work, so I isntalled everything with current verisons (reqquireemtns_no_versions.txt)*

- part 3: Tasks Done, deviated from plan
    - check #TODOs in `loader.py`
    - added additional mini tests
    - random seeds are not in one place
    - output of `cache_index`
            Each line is a JSON object with:
            - task_id: filename stem
            - n_train_pairs: number of train pairs
            - n_test_pairs: number of test pairs    
            - train_input_shapes: list of (rows, cols) for each train input
            - train_output_shapes: list of (rows, cols) for each train output
            - test_input_shapes: list of (rows, cols) for each test input
            - test_output_shapes: list of (rows, cols) for each test output (may be None)
    - outputs of `create_split_manifests` are in data/processedd folder, as dev_tasks.json and eval_tasks.json


- review part 3 with Vedant and Seun


## 10 8 25 - @caleb

Part 4
- implemented all off Part 4 with changes
- changes
    - i'm using np.int8 instead of np.uint8 in case there is signinficance to adding/subtracting palettes with results outside of 0 to 9

    - all seed in seeding.py

    - change in Part 4: made `palette` and `to_list` methods of Grid, as opposed to general functions

    - made `from_list` the `__init__` constructor

    - added multiple additional tests and Grid validation checks
- update previous parts
    - updated `loader.py`` with seeding from `seeding.py`
- other
    - made `run_tests.py` in `scripts/` that runs all test in `tests/` foler
