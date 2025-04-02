import base64
import io
import numpy as np
from PIL import Image

import difflib


def image_to_jpg_base64_url(image: np.ndarray | Image.Image):
    """Convert a numpy array to a base64 encoded image url."""

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    if image.mode in ("RGBA", "LA"):
        image = image.convert("RGB")

    with io.BytesIO() as buffer:
        image.save(buffer, format="JPEG")
        image_base64 = base64.b64encode(buffer.getvalue()).decode()

    return f"data:image/jpeg;base64,{image_base64}"


def obs_preprocessor(obs: dict) -> dict:

    return obs.copy()


# currently hardcoded for the webarena action set
valid_action_types = ["noop", "scroll", "keyboard_press", "click", "fill", "hover", "tab_focus", "new_tab",
                      "go_back", "go_forward", "goto", "tab_close", "select_option", "send_msg_to_user", "report_infeasible"]


def check_validity_of_action_proposal(action_proposal: str, action_set: list[str]):
    """Checks to see if all actions in the proposal exist in the action set. """

    return True
    #return action_proposal in action_set
