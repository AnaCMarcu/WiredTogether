import logging
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime
import torch
import numpy as np
from PIL import Image

# Biscuit environment depth dictionary
depth_dictionary = {
    "StoveBurner_7e32c1bb": 1,
    "StoveBurner_071647e1": 1,
    "StoveBurner_bd86687e": 1,
    "Cabinet_2fe8d304": 1,
    "CounterTop_cbb462c5": 1,
    "StoveBurner_5fa31b8b": 1,
    "CounterTop_a5da87e4": 1,
    "Cabinet_504a60f1": 1,
    "Potato_4a66b654": 1,
    "StoveKnob_9c0cf3b3": 1,
    "StoveKnob_43ec6a71": 1,
    "StoveKnob_5dfdfc01": 1,
    "StoveKnob_2dc524db": 1,
    "Pan_94d14525": 1,
    "Toaster_9125320d": 1,
    "Plate_41b068d4": 1,
    "Egg_0d9cfd77": 1,
    "Microwave_a03dc8f5": 1,
    # sliced items: potato, egg
    "Egg_Cracked_10(Clone)": 2,
    "Potato_10_Slice_1": 2,
    "Potato_10_Slice_2": 2,
    "Potato_10_Slice_3": 2,
    "Potato_10_Slice_4": 2,
    "Potato_10_Slice_5": 2,
    "Potato_10_Slice_6": 2,
    "Potato_10_Slice_7": 2,
    "Potato_10_Slice_8": 2,
}
# full kitchen plan depth dictionary
# depth_dictionary = {
#     "StoveBurner_90a47a45": 1,
#     "Drawer_5e387f7f": 1,
#     "Drawer_9d2b3ad1": 1,
#     "StoveBurner_ab505bab": 1,
#     "Drawer_18b84f39": 1,
#     "CounterTop_bafd4140": 1,
#     "Cabinet_deb040a8": 1,
#     "CounterTop_c83f4e59": 1,
#     "StoveBurner_b7865068": 1,
#     "Drawer_cbfc0458": 1,
#     "Drawer_23503746": 1,
#     "Cabinet_242ff8ff": 1,
#     "StoveKnob_0082d1cf": 1,
#     "Drawer_badab232": 1,
#     "StoveBurner_41e8e219": 1,
#     "Window_28489b94": 1,
#     "StoveKnob_8dd4e2fd": 1,
#     "Sink_1464872c": 1,
#     "StoveKnob_5e53d5ec": 1,
#     "Cabinet_2b789559": 1,
#     "Drawer_fb00b65a": 1,
#     "Cabinet_d332c18d": 1,
#     "Cabinet_60ab5497": 1,
#     "Cabinet_742b34de": 1,
#     "Floor_cc38bd03": 1,
#     "Cabinet_5e0161e9": 1,
#     "Drawer_62182f57": 1,
#     "StoveKnob_a95ae457": 1,
#     "Cabinet_f9038229": 1,
#     "Book_3d15d052": 1,
#     "Drawer_1d5beb62": 1,
#     "Cabinet_aa8596f8": 1,
#     "CounterTop_d7cc8dfe": 1,
#     "Glassbottle_a6112717": 1,
#     "Knife_9213a3e1": 2,
#     "Microwave_b200e0bc": 1,
#     "Bread_a13c4e42": 1,
#     "Fork_4fd82f45": 2,
#     "Shelf_62039e30": 1,
#     "Potato_2974d22d": 1,
#     "HousePlant_42ad2a02": 1,
#     "Toaster_6030edce": 1,
#     "SoapBottle_b17f0525": 1,
#     "Kettle_c1f85c6e": 1,
#     "Shelf_cb297b6a": 1,
#     "Pan_da081f05": 1,
#     "Plate_95e69397": 2,
#     "Tomato_7d6fd278": 1,
#     "Vase_2da5e7a5": 1,
#     "GarbageCan_dd6cf8f2": 1,
#     "Egg_cf1753df": 2,
#     "CreditCard_acee2f3e": 1,
#     "WineBottle_0e9c1ce8": 2,
#     "Pot_d7312b91": 1,
#     "Spatula_1e83453b": 1,
#     "PaperTowelRoll_a1e47ced": 1,
#     "Cup_8b4c228c": 2,
#     "Vase_d661eeca": 1,
#     "Shelf_cdcc01aa": 1,
#     "Fridge_4e5ce42a": 1,
#     "CoffeeMachine_ed06e3d1": 1,
#     "Bowl_e8075710": 2,
#     "SinkBasin_7db967d1": 1,
#     "SaltShaker_e142145f": 1,
#     "PepperShaker_e5dac52a": 1,
#     "Lettuce_b97186e2": 1,
#     "ButterKnife_cb56c43b": 1,
#     "Apple_34d5f204": 1,
#     "DishSponge_1be9f13b": 1,
#     "Spoon_b76dffe8": 2,
#     "LightSwitch_9dedd91c": 1,
#     "Mug_77db6e4d": 1,
#     "ShelvingUnit_4aa7ccd6": 1,
#     "Statue_150f16e9": 1,
#     "Stool_95215acf": 1,
#     "Stool_f121b870": 1,
#     "Faucet_198329de": 1,
#     # ^ initialised iems | items the agent creates \/
#     "Apple_1_Sliced_3": 3,
#     "Apple_1_Sliced_2": 3,
#     "Apple_1_Sliced_1": 3,
#     "Bread_1_Slice_9": 3,
#     "Bread_1_Slice_8": 3,
#     "Bread_1_Slice_7": 3,
#     "Bread_1_Slice_6": 3,
#     "Bread_1_Slice_5": 3,
#     "Bread_1_Slice_4": 3,
#     "Bread_1_Slice_3": 3,
#     "Bread_1_Slice_2": 3,
#     "Bread_1_Slice_1": 3,
#     "Tomato_1_Slice_7": 3,
#     "Tomato_1_Slice_6": 3,
#     "Tomato_1_Slice_5": 3,
#     "Tomato_1_Slice_4": 3,
#     "Tomato_1_Slice_3": 3,
#     "Tomato_1_Slice_2": 3,
#     "Tomato_1_Slice_1": 3,
#     "Potato_1_Slice_8": 3,
#     "Potato_1_Slice_7": 3,
#     "Potato_1_Slice_6": 3,
#     "Potato_1_Slice_5": 3,
#     "Potato_1_Slice_4": 3,
#     "Potato_1_Slice_3": 3,
#     "Potato_1_Slice_2": 3,
#     "Potato_1_Slice_1": 3,
#     "Lettuce_1_Slice_3": 3,
#     "Lettuce_1_Slice_2": 3,
#     "Lettuce_1_Slice_1": 3,
#     "Egg_Cracked_1(Clone)": 2,
# }


class Metric:
    def __init__(
        self,
        agent_type="None",
        prediction_type="None",
        number_of_agents=1,
        communication=True,
        path="./run_metrics",
    ):
        self.agent_type = agent_type
        self.prediction_type = prediction_type
        self.num_agents = number_of_agents
        self.communication = communication
        self.skills_learned = []
        self.novel_items = []
        self.timestep = 0
        self.saved_timesteps = {
            "timesteps": [],
            "num_skills_learned": [],
            "num_novel_items": [],
            "exploration_score": [],
        }
        if number_of_agents > 1:
            self.skills_learned_joint = []
            self.novel_items_joint = []
            self.saved_timesteps["num_skills_learned_joint"] = []
            self.saved_timesteps["num_novel_items_joint"] = []
            self.saved_timesteps["exploration_score_joint"] = []
        self.interventions_found = []
        self.depth_dictionary = depth_dictionary
        self.target_folder = self.mkdir_metrics(path=path)

    def found_skill(self, description: str, main=True):
        if (
            description.lower() not in [s[1].lower() for s in self.skills_learned]
            and main
        ):
            self.skills_learned.append((self.timestep, description))
        if self.num_agents > 1 and description.lower() not in [
            s[1].lower() for s in self.skills_learned_joint
        ]:
            self.skills_learned_joint.append((self.timestep, description))

    def found_items(self, item: str, main=True):
        unique_items = [i[1] for i in self.novel_items]
        if item not in unique_items and main:
            self.novel_items.append((self.timestep, item))
        if self.num_agents > 1 and item not in [i[1] for i in self.novel_items_joint]:
            self.novel_items_joint.append((self.timestep, item))

    def calcDepth(self, item: str) -> int:
        if item.split(" ")[0] in self.depth_dictionary:
            depth = self.depth_dictionary[item.split(" ")[0]]
        else:
            depth = 1
            logging.warning(
                f"Item <{item.split(' ')[0]}> not found in depth dictionary, defaulting to depth 1"
            )
        for prop in [
            " filled",
            " broken",
            " dirty",
            " cooked",
            " usedUp",
            " Hot",
            " Cold",
        ]:
            if prop in item:
                depth += 1
        if "containing" in item and "counter" not in item:
            # counter starts with containing all items so dont count that
            num_items = len(item.split("containing")[1].split(","))
            depth += num_items
        return depth

    def store_timestep(self):
        self.saved_timesteps["timesteps"].append(self.timestep)
        self.saved_timesteps["num_skills_learned"].append(len(self.skills_learned))

        # if new item found this
        if not self.novel_items:
            self.saved_timesteps["num_novel_items"].append(0)
        else:
            curr_novel_items = 0
            if self.saved_timesteps["num_novel_items"]:
                curr_novel_items = self.saved_timesteps["num_novel_items"][-1]
            if self.novel_items[-1][0] == self.timestep:
                # found a new item this timestep
                item_depth = self.calcDepth(self.novel_items[-1][1])
                self.saved_timesteps["num_novel_items"].append(
                    curr_novel_items + item_depth
                )
            else:
                # did not find a new item this timestep
                self.saved_timesteps["num_novel_items"].append(curr_novel_items)

        self.saved_timesteps["exploration_score"].append(
            self.saved_timesteps["num_skills_learned"][-1]
            + self.saved_timesteps["num_novel_items"][-1]
        )
        if self.num_agents > 1:
            self.saved_timesteps["num_skills_learned_joint"].append(
                len(self.skills_learned_joint)
            )

            # if new item found this
            if not self.novel_items_joint:
                self.saved_timesteps["num_novel_items_joint"].append(0)
            else:
                curr_novel_items_joint = 0
                if self.saved_timesteps["num_novel_items_joint"]:
                    curr_novel_items_joint = self.saved_timesteps[
                        "num_novel_items_joint"
                    ][-1]
                # multiple agents can find different items in the same timestep
                curr_new_joint_items = [
                    (t, item)
                    for t, item in self.novel_items_joint
                    if t == self.timestep
                ]

                curr_found_depth = 0
                for t, item in curr_new_joint_items:
                    curr_found_depth += self.calcDepth(item)

                self.saved_timesteps["num_novel_items_joint"].append(
                    curr_novel_items_joint + curr_found_depth
                )

            self.saved_timesteps["exploration_score_joint"].append(
                self.saved_timesteps["num_skills_learned_joint"][-1]
                + self.saved_timesteps["num_novel_items_joint"][-1]
            )

        self.timestep += 1

    def mkdir_metrics(self, path="./", folder_name=None, file_name="data.json"):
        # Ensure base directory exists
        
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

        if folder_name is None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            if self.communication:
                folder_name = f"metrics_{self.agent_type}_{self.num_agents}agents_comm_pred:{self.prediction_type}_{timestamp}"
            else:
                folder_name = f"metrics_{self.agent_type}_{self.num_agents}agents_noComm__pred:{self.prediction_type}_{timestamp}"

        # Create target folder
        target_folder = os.path.join(path, folder_name)
        os.makedirs(target_folder, exist_ok=True)

        # if agent_type

        return target_folder

    def save_run_metrics(self, file_name="data.json"):

        file_path = os.path.join(self.target_folder, file_name)

        # concate data to dictionary
        self.dictionary = {
            "skills_learned": self.skills_learned,
            "novel_items": self.novel_items,
            "interventions_found": self.interventions_found,
            "saved_timesteps": self.saved_timesteps,
        }

        if self.num_agents > 1:
            self.dictionary["skills_learned_joint"] = self.skills_learned_joint
            self.dictionary["novel_items_joint"] = self.novel_items_joint

        # Save dictionary as JSON
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.dictionary, f, indent=4, ensure_ascii=False)

        self.save_graphs(path=self.target_folder)
        return file_path

    def save_graphs(self, path="./"):

        if not os.path.exists(path):
            os.makedirs(path)

        plt.figure()
        plt.plot(
            self.saved_timesteps["timesteps"],
            self.saved_timesteps["num_skills_learned"],
            label="Skills Learned",
        )
        plt.plot(
            self.saved_timesteps["timesteps"],
            self.saved_timesteps["num_novel_items"],
            label="Novel Items",
        )
        plt.plot(
            self.saved_timesteps["timesteps"],
            self.saved_timesteps["exploration_score"],
            label="Exploration Score",
        )
        plt.xlabel("Timesteps")
        plt.ylabel("Counts")
        plt.title("Exploration Metrics Over Time")
        plt.legend()
        plt.savefig(os.path.join(path, "exploration_metrics.png"))
        plt.close()

        if self.num_agents > 1:
            plt.figure()
            plt.plot(
                self.saved_timesteps["timesteps"],
                self.saved_timesteps["num_skills_learned_joint"],
                label="Joint Skills Learned",
            )
            plt.plot(
                self.saved_timesteps["timesteps"],
                self.saved_timesteps["num_novel_items_joint"],
                label="Joint Novel Items",
            )
            plt.plot(
                self.saved_timesteps["timesteps"],
                self.saved_timesteps["exploration_score_joint"],
                label="Joint Exploration Score",
            )
            plt.xlabel("Timesteps")
            plt.ylabel("Counts")
            plt.title("Joint Exploration Metrics Over Time")
            plt.legend()
            plt.savefig(os.path.join(path, "joint_exploration_metrics.png"))
            plt.close()

    def log(self, text, filepath="log.txt"):
        full_path = os.path.join(self.target_folder, filepath)

        # Open the file in append mode, create it if it doesn't exist
        with open(full_path, "a") as f:
            f.write(text + "\n")

    # ---
    def save_predictions(
        self,
        original_image,
        predicted_image,
        enc_dec_image,
        action,
        held_item=None,
        path="./predictions",
        extra_info=None,
        mapper=None,
    ):
        interaction_causal_vars = mapper(extra_info["interaction_indices"])

        full_path = os.path.join(self.target_folder, path)
        if not os.path.exists(full_path):
            os.makedirs(full_path)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Visualize
        # Create a single figure with 3 columns
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Original image
        axes[0].imshow(original_image)
        axes[0].set_title("Original")
        axes[0].axis("off")

        # Enc-Dec image
        axes[1].imshow(enc_dec_image)
        axes[1].set_title("Enc-Dec")
        axes[1].axis("off")

        # Predicted image
        axes[2].imshow(predicted_image)
        axes[2].set_title("Predicted")
        axes[2].axis("off")

        plt.suptitle(f"Action: {action}, Held Item: {held_item}", fontsize=16)
        # Adjust layout so caption doesn’t overlap
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # leave space at the bottom

        format_string = "Causal variables interacted with: "
        for key, value in interaction_causal_vars.items():
            if not len(value) == 0:
                format_string += f"{key} {value}, "

        # ✨ Add caption at bottom center
        fig.text(
            0.5,  # x position (0=left, 1=right)
            0.01,  # y position (0=bottom, 1=top)
            format_string,  # caption text
            ha="center",  # horizontal alignment
            va="bottom",  # vertical alignment
            fontsize=10,
            color="gray",
        )

        plt.savefig(
            os.path.join(full_path, f"predictions_{timestamp}_{action}_{held_item}.png")
        )
        plt.close()

        prior_interaction_path = os.path.join(full_path, "./prior_interaction")

        if not os.path.exists(prior_interaction_path):
            os.makedirs(prior_interaction_path)

        # Create one figure with 1 row and 2 columns
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # Adjust figsize as needed

        # Plot first figure on the first subplot
        self.plot_interaction_strength(
            extra_info=extra_info, action_text=action, ax=axes[0]
        )
        axes[0].set_title(
            "Interaction Strength"
        )  # Optional: set a title for this subplot

        # Plot second figure on the second subplot
        self.plot_prior(extra_info=extra_info, title_prefix="Prior", ax=axes[1])
        axes[1].set_title("Prior Distribution")  # Optional

        # Adjust layout to prevent overlap
        plt.tight_layout()

        # Save as a single figure
        fig.savefig(
            os.path.join(
                prior_interaction_path, f"combined_{timestamp}_{action}_{held_item}.png"
            )
        )
        plt.close(fig)

        raw_data_path = os.path.join(full_path, "./raw_data")

        # 1. Ensure directory exists
        os.makedirs(raw_data_path, exist_ok=True)

        # 2. Timestamp for unique filenames
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Helper to save an image safely
        def _save_image(img, filename):
            filepath = os.path.join(raw_data_path, filename)
            plt.imsave(filepath, img)
            return filepath

        # 3. Save images
        orig_path = _save_image(
            original_image,
            f"{self.timestep}_{timestamp}_original_{action}_{held_item}.png",
        )
        pred_path = _save_image(
            predicted_image,
            f"{self.timestep}_{timestamp}_prediction_{action}_{held_item}.png",
        )
        encdec_path = _save_image(
            enc_dec_image,
            f"{self.timestep}_{timestamp}_encdec_{action}_{held_item}.png",
        )

    def save_surgical_interventions(
        self,
        original_image,
        original_encdec,
        original_prediction,
        original_interv_encdec,
        intervention_action,
        intervention_prediction,
        intervention_encdec,
        action,
        final_prediction,
        held_item=None,
        predicted_held_item=None,
        original_extra_info=None,
        candidate_extra_info=None,
        final_extra_info=None,
        mapper=None,
        latent_idx=None,
        path="./surgical_interventions",
        save_raw_data=False,
    ):
        original_interaction_causal_vars = mapper(
            original_extra_info["interaction_indices"]
        )
        candidate_interaction_causal_vars = mapper(
            candidate_extra_info["interaction_indices"]
        )
        prediction_interaction_causal_vars = mapper(
            final_extra_info["interaction_indices"]
        )

        full_path = os.path.join(self.target_folder, path)
        os.makedirs(full_path, exist_ok=True)
        full_path = os.path.join(full_path, str(self.timestep))
        os.makedirs(full_path, exist_ok=True)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        fig, axes = plt.subplots(3, 3, figsize=(18, 18))
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.margins(0)

        # ✅ Row 1: Original rollout
        axes[0, 0].imshow(original_image)
        axes[0, 0].set_title("Original")
        axes[0, 0].axis("off")

        axes[0, 1].imshow(original_encdec)
        axes[0, 1].set_title("Original Enc-Dec")
        axes[0, 1].axis("off")

        axes[0, 2].imshow(original_prediction)
        axes[0, 2].set_title("Original Prediction")
        axes[0, 2].axis("off")

        # ✅ Row 2: Intervention rollout
        axes[1, 0].imshow(original_image)
        axes[1, 0].set_title("Original (again)")
        axes[1, 0].axis("off")

        axes[1, 1].imshow(original_interv_encdec)
        axes[1, 1].set_title("Original Enc-Dec for Intervention")
        axes[1, 1].axis("off")

        axes[1, 2].imshow(intervention_prediction)
        axes[1, 2].set_title("Intervention Prediction")
        axes[1, 2].axis("off")

        # ✅ Row 3: Final predicted rollout
        axes[2, 0].imshow(intervention_prediction)
        axes[2, 0].set_title("Intervention Prediction (Again)")
        axes[2, 0].axis("off")

        axes[2, 1].imshow(intervention_encdec)
        axes[2, 1].set_title("Intervention Prediction Enc-Dec")
        axes[2, 1].axis("off")

        axes[2, 2].imshow(final_prediction)
        axes[2, 2].set_title("Final Prediction")
        axes[2, 2].axis("off")

        plt.suptitle(
            f"Intervention: {intervention_action}, Base Action: {action}, Held Item: {held_item}, Predicted Held Item: {predicted_held_item}",
            fontsize=18,
        )

        # plt.tight_layout(rect=[0, 0.18, 1, 0.92])
        plt.tight_layout(pad=3, h_pad=5)

        # Helper to format row captions
        def format_vars(prefix, vars_dict):
            if vars_dict:
                vals = [f"{k} {v}" for k, v in vars_dict.items() if v]
                return prefix + (", ".join(vals) if vals else "None")
            return prefix + "None"

        # ✅ Captions below each row
        fig.text(
            0.5,
            0.65,
            format_vars(
                "Original interaction vars: ", original_interaction_causal_vars
            ),
            ha="center",
            fontsize=11,
            color="gray",
        )

        fig.text(
            0.5,
            0.33,
            format_vars(
                "Candidate interaction vars: ", candidate_interaction_causal_vars
            ),
            ha="center",
            fontsize=11,
            color="gray",
        )

        fig.text(
            0.5,
            0.01,
            format_vars(
                "Prediction interaction vars: ", prediction_interaction_causal_vars
            ),
            ha="center",
            fontsize=11,
            color="gray",
        )

        # ✅ Save figure
        filename = (
            f"surg_interventions_{timestamp}_{self.timestep}_{action}_{held_item}.png"
        )

        plt.savefig(os.path.join(full_path, filename))
        plt.close()

        # save intervention
        # save prior
        prior_interaction_path = os.path.join(full_path, "./prior_interaction")

        if not os.path.exists(prior_interaction_path):
            os.makedirs(prior_interaction_path, exist_ok=True)

        # Create figure: 3 rows × 2 columns
        fig, axes = plt.subplots(3, 2, figsize=(14, 16))

        # Row 1 — Base
        self.plot_interaction_strength(
            original_extra_info, action_text=action, ax=axes[0, 0]
        )
        self.plot_prior(original_extra_info, ax=axes[0, 1], title_prefix="Base Prior")

        # Row 2 — Intervention
        self.plot_interaction_strength(
            candidate_extra_info, action_text=intervention_action, ax=axes[1, 0]
        )

        self.plot_prior(
            candidate_extra_info, ax=axes[1, 1], title_prefix="Intervention Prior"
        )

        # Row 3 — Prediction
        self.plot_interaction_strength(
            final_extra_info, action_text=action, ax=axes[2, 0]
        )

        self.plot_prior(
            final_extra_info, ax=axes[2, 1], title_prefix="Prediction Prior"
        )

        # Layout improvements
        plt.tight_layout(pad=3.5)
        plt.suptitle(
            f"Base: {action} | Intervention: {intervention_action} | Latent Index: {latent_idx}|"
            f"Held: {held_item} | Predicted Held: {predicted_held_item}",
            fontsize=16,
            y=0.995,
        )

        filename = (
            f"intervention_prior_graphs_{timestamp}_{self.timestep}_"
            f"{action}_{intervention_action}.png"
        )

        plt.savefig(os.path.join(prior_interaction_path, filename), bbox_inches="tight")
        plt.close()

        if save_raw_data:
            raw_data_path = os.path.join(full_path, "./raw_data")

            if not os.path.exists(raw_data_path):
                os.makedirs(raw_data_path, exist_ok=True)

            def safe_convert(value):
                if isinstance(value, torch.Tensor):
                    return value.detach().cpu().numpy().tolist()
                elif isinstance(value, np.ndarray):
                    return value.tolist()
                elif isinstance(value, (int, float, str, bool)) or value is None:
                    return value
                elif isinstance(value, dict):
                    return {k: safe_convert(v) for k, v in value.items()}
                elif isinstance(value, (list, tuple)):
                    return [safe_convert(v) for v in value]
                else:
                    return str(value)

            def save_image(name, img_data):
                if isinstance(img_data, torch.Tensor):
                    img_data = img_data.detach().cpu().numpy()
                if img_data.ndim == 3 and img_data.shape[0] in [1, 3]:  # CHW → HWC
                    img_data = np.transpose(img_data, (1, 2, 0))
                img_data = (img_data * 255).clip(0, 255).astype(np.uint8)
                img_path = os.path.join(raw_data_path, f"{name}.png")
                Image.fromarray(img_data).save(img_path)
                return img_path

            image_paths = {
                "original_image": save_image("original_image", original_image),
                "original_encdec": save_image("original_encdec", original_encdec),
                "original_prediction": save_image(
                    "original_prediction", original_prediction
                ),
                "original_interv_encdec": save_image(
                    "original_interv_encdec", original_interv_encdec
                ),
                "intervention_prediction": save_image(
                    "intervention_prediction", intervention_prediction
                ),
                "intervention_encdec": save_image(
                    "intervention_encdec", intervention_encdec
                ),
                "final_prediction": save_image("final_prediction", final_prediction),
            }

            data = {
                "metadata": {
                    "timestamp": timestamp,
                    "action": action,
                    "intervention_action": intervention_action,
                    "held_item": held_item,
                    "predicted_held_item": predicted_held_item,
                    "latent_idx": latent_idx,
                },
                "image_paths": image_paths if save_raw_data else "Not saved",
                "original_extra_info": safe_convert(original_extra_info),
                "candidate_extra_info": safe_convert(candidate_extra_info),
                "final_extra_info": safe_convert(final_extra_info),
            }
            raw_data_json_path = os.path.join(raw_data_path, f"data_{timestamp}.json")
            with open(raw_data_json_path, "w") as f:
                json.dump(data, f, indent=4)

            return full_path

    def plot_interaction_strength(self, extra_info, action_text=None, ax=None):
        """Plots interaction strength bar chart into provided axis."""

        # Extract interaction strength vector
        interaction_strength = extra_info["action"].detach().squeeze(-1).squeeze(0)

        # Use provided axis or create new figure
        own_fig = False
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 4))
            own_fig = True

        values = interaction_strength.cpu().numpy()
        ax.bar(range(len(values)), values)

        ax.set_xlabel("Latent Index")
        ax.set_ylabel("Influence Strength")

        title = "Latent Action Influence"
        if action_text is not None:
            title += f" ({action_text})"
        ax.set_title(title)

        # If we created our own figure, return it
        if own_fig:
            return fig, ax
        return ax

    def plot_prior(self, extra_info, ax=None, title_prefix="Prior"):
        """
        Plot prior mean values. If `ax` provided, draw inside that subplot.
        Otherwise return a standalone figure + axis.
        """
        prior_mean, prior_logstd = extra_info["prior"]

        # Convert to NumPy and flatten to 1D
        prior_mean_npy = prior_mean.detach().cpu().reshape(-1).numpy()
        prior_std_npy = np.exp(prior_logstd.detach().cpu().reshape(-1).numpy())

        N = len(prior_mean_npy)
        x = np.arange(N)

        own_fig = False
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 4))
            own_fig = True

        # Plot bars with std error bars
        ax.bar(x, prior_mean_npy, yerr=prior_std_npy, ecolor="black", capsize=3)

        ax.set_xlabel("Latent Index")
        ax.set_ylabel("Mean ± Std")
        ax.set_title(f"{title_prefix} Mean & Std")
        ax.set_xlim([-1, N])

        if own_fig:
            plt.tight_layout()
            return fig, ax

        return ax

    # state of env actually does not matter as if microwave door is open or closed, opening/closing it will change the outcome (work/wont work)
    # Both are surgical interventions:
    # open door if closed -> put plate in microwave (will work instead)
    # close door if open -> take plate out of microwave (will not work instead)
    def check_surgical(self, action, held_item, valid_interventions=None):
        possible = False
        description = ""
        # put plate in microwave -> open door -> ...
        if (
            held_item is not None
            and "plate" in held_item.lower()
            and "putobject" in action.lower()
            and "microwave" in action.lower()
        ):
            possible = True
            description = "Put plate in microwave"
        # put egg in microwave ->open door -> ...
        elif (
            held_item is not None
            and "egg" in held_item.lower()
            and "putobject" in action.lower()
            and "microwave" in action.lower()
        ):
            possible = True
            description = "Put egg in microwave"
        # put egg in pan -> toggle stove -> ...
        elif (
            held_item is not None
            and "egg" in held_item.lower()
            and "putobject" in action.lower()
            and "pan" in action.lower()
        ):
            possible = True
            description = "Put egg in pan"
        # toggle microwave -> close door -> ...
        elif "toggle" in action.lower() and "microwave" in action.lower():
            possible = True
            description = "Toggle microwave door"
        # toggle stove -> put pan with egg on stove ->...
        elif (
            held_item is not None
            and "pan" in held_item.lower()
            and "egg" in held_item.lower()
            and "putobject" in action.lower()
            and "stove" in action.lower()
        ):
            possible = True
            description = "Put egg in pan on stove"

        if possible:
            self.log(
                f"A surgical intervention is possible for action {action}: {possible}, {description}"
            )
            if not valid_interventions or len(valid_interventions) == 0:
                self.interventions_found.append(
                    (self.timestep, description, None, action)
                )
            else:
                self.interventions_found.append(
                    (self.timestep, description, valid_interventions[0][0], action)
                )

        return possible, description
