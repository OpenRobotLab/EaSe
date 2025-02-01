import base64
import io
from PIL import Image

def resize_image_to_GPT_size(image, detail="high"):
    max_size = 2048
    min_size = 768
    image_copy = image.copy()

    width, height = image_copy.size
    aspect_ratio = width / height

    if detail == "high":
        if max(width, height) > max_size:
            if width > height:
                width = max_size
                height = int(width / aspect_ratio)
            else:
                height = max_size
                width = int(height * aspect_ratio)

            image_copy = image_copy.resize((width, height), Image.LANCZOS)

        if min(width, height) > min_size:
            if width > height:
                height = min_size
                width = int(height * aspect_ratio)
            else:
                width = min_size
                height = int(width / aspect_ratio)

            image_copy = image_copy.resize((width, height), Image.LANCZOS)
    else:
        print("resize_image_to_GPT_size: Unsupport mode:", detail)

    return image_copy

def encode_PIL_image_to_base64(image):
    # Save the image to a bytes buffer
    buf = io.BytesIO()
    image.save(buf, format="JPEG")

    # Get the byte data from the buffer
    byte_data = buf.getvalue()

    # Encode the byte data to base64
    base64_str = base64.b64encode(byte_data).decode("utf-8")
    return base64_str

user_prompt = """Imagine you are in a room and are asked to find one object. Given a series of images from a video scanning an indoor room and a query describing a specific object in the room, you need to analyze the images to find the object that matchs in the description best from some candidates according to images. 

You will be provided an image consisting of small images in grid layout. In each small image, the candidates in the image are annotated with their ID like "obj_id" in red text. Do not mix the annotated IDs with the actual appearance of the objects. Your task is to give me id of  the most possible object based on the description. 

Notice that:
1. You MUST ONLY check the matching between object in the red bounding box and the description. For example, if both image A and B show the matching object, but only in image A, the matching object is within the bounding box, you should pick A.
2. The predicted label in the top-left corner is not always correct. You should also judge the object based on the image content. If the predicted label is obviously different from the object in bounding box, correct by your self.
3. Your response should be formatted as a JSON object with three keys "reasoning", "answer" like this:

{{
    "reasoning": "your reasons", // Explain the justification why you select the object ID.
    "object id": id // An integer. The object ID you selected. Always give one object ID from the image, which you are the
    most confident of, even you think the image does not contain the correct object.
}}

Now start the task:
Target Object Description: "{utterance}"
Candidate list: {candidates}
"""