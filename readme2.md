Two folders: evaluation and Training


Each task has
    - train pairs 
    - test pairs


NOTE: both Training and Evaluation sets have "train pairs"


For a Tasks, the train and test examles can all have ddifferent sizes

![[image.png]]

we *treat* loaded grids as read-only and copy when transforming. This prevents accidental in-place edits during search or scoring.