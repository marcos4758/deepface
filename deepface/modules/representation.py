# built-in dependencies
from typing import Any, Dict, List, Union, Optional, IO

# 3rd party dependencies
import numpy as np
import cv2

# project dependencies
from deepface.commons import image_utils
from deepface.modules import modeling, detection, preprocessing
from deepface.models.FacialRecognition import FacialRecognition


def represent(
    img_path: Union[str, np.ndarray, IO[bytes]],
    model_name: str = "VGG-Face",
    enforce_detection: bool = True,
    detector_backend: str = "opencv",
    align: bool = True,
    expand_percentage: int = 0,
    normalization: str = "base",
    color_face: str = "rgb",
    normalize_face: bool = True,
    anti_spoofing: str = "skip",
    max_faces: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Represent facial images as multi-dimensional vector embeddings.

    Args:
        img_path (str or np.ndarray): The exact path to the image, a numpy array in BGR format,
            or a base64 encoded image. If the source image contains multiple faces, the result will
            include information for each detected face.

        model_name (str): Model for face recognition. Options: VGG-Face, Facenet, Facenet512,
            OpenFace, DeepFace, DeepID, Dlib, ArcFace, SFace and GhostFaceNet

        enforce_detection (boolean): If no face is detected in an image, raise an exception.
            Default is True. Set to False to avoid the exception for low-resolution images.

        detector_backend (string): face detector backend. Options: 'opencv', 'retinaface',
            'mtcnn', 'ssd', 'dlib', 'mediapipe', 'yolov8', 'yolov11n', 'yolov11s',
            'yolov11m', 'centerface' or 'skip'.

        align (boolean): Perform alignment based on the eye positions.

        expand_percentage (int): expand detected facial area with a percentage (default is 0).

        normalization (string): Normalize the input image before feeding it to the model.
            Default is base. Options: base, raw, Facenet, Facenet2018, VGGFace, VGGFace2, ArcFace

        color_face (string): Color to return face image output. Options: 'rgb', 'bgr' or 'gray'
            (default is 'rgb').

        normalize_face (boolean): Flag to enable normalization (divide by 255) of the output
            face image output face image normalization (default is True).

        anti_spoofing (string): anti-spoofing analyze mode. Options: 'skip', 'run' or 'run_and_raise'.
            If 'run', the model will analyze the input image for spoofing. If 'run_and_raise',
            the model will also raise an exception if a spoof is detected (default is 'skip').

        max_faces (int): Set a limit on the number of faces to be processed (default is None).

    Returns:
        results (List[Dict[str, Any]]): A list of dictionaries, each containing the
            following fields:

        - "embedding" (List[float]): Multidimensional vector representing facial features.
            The number of dimensions varies based on the reference model
            (e.g., FaceNet returns 128 dimensions, VGG-Face returns 4096 dimensions).

        - "face" (np.ndarray): The detected face as a NumPy array. The color of the face
            is determined by the `color_face` parameter.

        - "facial_area" (Dict[str, Any]): The detected face's regions as a dictionary containing:
            - keys 'x', 'y', 'w', 'h' with int values
            - keys 'left_eye', 'right_eye' with a tuple of 2 ints as values.
                left eye and right eye are eyes on the left and right respectively with respect
                to the person itself instead of observer.

        - "face_confidence" (float): Confidence score of face detection. If `detector_backend` is set
            to 'skip', the confidence will be 0 and is nonsensical.

        - "is_real" (boolean): antispoofing analyze result. this key is just available in the
            result only if anti_spoofing is set to 'run' or 'run_and_raise' in input arguments.

        - "antispoof_score" (float): score of antispoofing analyze result. this key is
            just available in the result only if anti_spoofing is set to 'run' or 'run_and_raise'
            in input arguments.

        - "antispoof_prediction" (np.ndarray): prediction vector of antispoofing analyze result.
            this key is just available in the result only if anti_spoofing is set to 'run' or
            'run_and_raise' in input arguments.
    """
    resp_objs = []

    model: FacialRecognition = modeling.build_model(
        task="facial_recognition", model_name=model_name
    )

    target_size = model.input_shape

    # ---------------------------------
    # we have run pre-process in verification. so, this can be skipped if it is coming from verify.
    if detector_backend != "skip":
        img_objs = detection.extract_faces(
            img_path=img_path,
            detector_backend=detector_backend,
            enforce_detection=enforce_detection,
            align=align,
            expand_percentage=expand_percentage,
            grayscale=False,
            color_face="bgr",
            normalize_face=False,
            anti_spoofing=True if "run" in anti_spoofing else False,
            max_faces=max_faces,
        )
    else:  # skip
        # Try load. If load error, will raise exception internal
        img, _ = image_utils.load_image(img_path)

        if len(img.shape) != 3:
            raise ValueError(f"Input img must be 3 dimensional but it is {img.shape}")

        # Check if the image is normalized
        if img.max() <= 1:
            img = img * 255

        # make dummy region and confidence to keep compatibility with `extract_faces`
        img_objs = [
            {
                "face": img,
                "facial_area": {"x": 0, "y": 0, "w": img.shape[0], "h": img.shape[1]},
                "confidence": 0,
            }
        ]
    # ---------------------------------

    if max_faces is not None and max_faces < len(img_objs):
        # sort as largest facial areas come first
        img_objs = sorted(
            img_objs,
            key=lambda img_obj: img_obj["facial_area"]["w"] * img_obj["facial_area"]["h"],
            reverse=True,
        )
        # discard rest of the items
        img_objs = img_objs[0:max_faces]

    for img_obj in img_objs:

        is_real = img_obj.get("is_real", None)
        antispoof_score = img_obj.get("antispoof_score", None)
        antispoof_prediction = img_obj.get("antispoof_prediction", None)

        if anti_spoofing == "run_and_raise" and is_real is False:
            raise ValueError("Spoof detected in the given image.")

        face = img_obj["face"]
        facial_area = img_obj["facial_area"]
        face_confidence = img_obj["confidence"]

        if color_face == "rgb":
            face = face[:, :, ::-1]
        elif color_face == "bgr":
            pass  # image is in BGR
        elif color_face == "gray":
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError(f"The color_face can be rgb, bgr or gray, but it is {color_face}.")

        if normalize_face:
            face = face / 255  # normalize input in [0, 1]

        input_img = img_obj["face"]
        input_img = input_img / 255  # normalize input in [0, 1]

        # resize to expected shape of ml model
        input_img = preprocessing.resize_image(
            img=input_img,
            # thanks to DeepId (!)
            target_size=(target_size[1], target_size[0]),
        )

        # custom normalization
        input_img = preprocessing.normalize_input(img=input_img, normalization=normalization)

        embedding = model.forward(input_img)

        output = {
            "embedding": embedding,
            "face": face,
            "facial_area": facial_area,
            "face_confidence": face_confidence,
        }

        if "run" in anti_spoofing:
            output["is_real"] = is_real
            output["antispoof_score"] = antispoof_score
            output["antispoof_prediction"] = antispoof_prediction

        resp_objs.append(output)

    return resp_objs
