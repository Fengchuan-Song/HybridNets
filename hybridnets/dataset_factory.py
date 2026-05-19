def get_dataset_class(params):
    dataset_type = (params.dataset.get('type') or 'bdd100k').lower()
    if dataset_type in {'waterscenes', 'water_scenes'}:
        from hybridnets.waterscenes_dataset import WaterScenesDataset
        return WaterScenesDataset
    if dataset_type == 'custom':
        from hybridnets.custom_dataset import CustomDataset
        return CustomDataset
    from hybridnets.dataset import BddDataset
    return BddDataset
