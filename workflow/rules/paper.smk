# The paper (paper/workflow/Snakefile), a module of the one workflow:
# targets `development`, `paper` and `render` (workflow/README.md,
# section 7).


module paper_workflow:
    snakefile:
        "../../paper/workflow/Snakefile"
    config:
        config


use rule * from paper_workflow
