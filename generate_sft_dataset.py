from train_agent_with_rl import AlfworldEnv

env = AlfworldEnv(dataset_size=3000)
dataset = env.get_sft_dataset()
dataset.save_to_disk(dataset_path='./datasets/alphahome-sft')