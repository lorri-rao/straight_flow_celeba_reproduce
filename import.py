# wandb import script
# credit to https://github.com/wandb/wandb/issues/4790#issuecomment-2162968432
import wandb
wandb.login()
api = wandb.Api()

src_entity = "lodestone-rock"
src_project = "straight_flow_celeba"
# src_name = ""

dst_entity = "z-y00-amd"
dst_project = "straight_flow_celeba_reproduce"

runs = api.runs(f"{src_entity}/{src_project}")

for run in runs:
    print(run.name)
for run in runs:
    if "h100" in run.name:
        # Get the run history and files
        history = run.history()
        files = run.files()

        # Create a new run in the destination project
        new_run = wandb.init(project=dst_project, entity=dst_entity, config=run.config, name=run.name,resume="allow")
        
        # Log the history to the new run
        for index, row in history.iterrows():

            # By default copies over the wrong step size for me, so include the run's step_size. Can also enter this manually.
            step_size = history['_step'].values[1]
            
            new_run.log(row.to_dict(), step= index * step_size)

        # Upload the files to the new run
        for file in files:
            file.download(replace=True)
            new_run.save(file.name,policy = "now")

        # Finish the new run
        new_run.finish()