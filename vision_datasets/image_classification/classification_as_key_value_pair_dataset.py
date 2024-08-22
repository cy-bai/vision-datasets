import logging
import random
import typing
from abc import ABC, abstractmethod
from copy import deepcopy

from ..common import DatasetManifest, DatasetTypes, ImageDataManifest, KeyValuePairDatasetInfo
from ..common.dataset.base_dataset import VisionDataset
from ..common.dataset.vision_dataset import LocalFolderCacheDecorator, VisionDataset
from ..image_classification.manifest import ImageClassificationLabelManifest
from .manifest import KeyValuePairDatasetManifest, KeyValuePairLabelManifest

logger = logging.getLogger(__name__)


class ClassificationAsKeyValuePairDataset(VisionDataset):
    """Dataset class that access Classification datset as KeyValuePair dataset."""
    
    def __init__(self, classification_dataset: VisionDataset):
        """

        """

        if classification_dataset is None:
            raise ValueError
        if classification_dataset.dataset_info.dataset_type not in {DatasetTypes.IMAGE_CLASSIFICATION_MULTICLASS, DatasetTypes.IMAGE_CLASSIFICATION_MULTILABEL}:
            raise ValueError

        # Generate schema and update dataset info
        classification_dataset = deepcopy(classification_dataset)
        dataset_info_dict = classification_dataset.dataset_info.__dict__
        dataset_manifest = classification_dataset.dataset_manifest 
        schema = self.construc_schema(dataset_manifest.categories)
        dataset_info = KeyValuePairDatasetInfo(dataset_info_dict + {"schema": schema})
        
        # Update dataset manifest
        annotations = []
        for id, img in enumerate(dataset_manifest.images, 1):
            class_names = [dataset_manifest.categories[l.category_id] for l in img.labels]
            KeyValuePairLabelManifest(id, )
            annotations.append(DatasetManifestWithMultiImageLabel())
        annotations = [dataset_manifest.categories[l.category_id] for img in dataset_manifest.images for l in img.labels]
        dataset_manifest = KeyValuePairDatasetManifest(dataset_manifest.images, annotations, schema)
        super().__init__(dataset_info, dataset_manifest, dataset_resources=classification_dataset.dataset_resources)
        
    def construc_schema(self, class_names):
        if self.dataset_info.dataset_type == DatasetTypes.IMAGE_CLASSIFICATION_MULTICLASS:
            schema = {
                "name": "Multiclass image classification",
                "description": f"Classify images into one of the provided classes.",
                "fieldSchema": {
                    "className": {
                        "type": "string",
                        "description": "Class name that the image belongs to.",
                        "classes": {c: {} for c in class_names}
                    }
                }
            }
        else:
            schema = {
                "name": "Multilabel image classification",
                "description": f"Classify images into one or more of the provided classes.",
                "fieldSchema": {
                    "className": {
                        "type": "array",
                        "description": "All the class names that the image belongs to.",
                        "items": {
                            "type": "string",
                            "description": "The name of each class.",
                            "classes": {c: {} for c in class_names}
                        }
                    }
                }
            }
        return schema
