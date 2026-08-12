import argparse

from dataset.mesh_dataset import Teeth3DSDataset

CACHE_CONFIGS = {
    "capped16k": dict(processed_folder="processed_capped16k", target_density=5.0, max_target_count=16000),
    "w5": dict(processed_folder="processed_w5", target_density=5.0, max_target_count=None),
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a Teeth3DS density-downsampled cache")
    parser.add_argument("--cache", choices=list(CACHE_CONFIGS), required=True,
                         help="Which cache config to build: 'capped16k' (legacy full-arch path, "
                              "bounded N) or 'w5' (uncapped, true W=5 density everywhere)")
    args = parser.parse_args()

    config = CACHE_CONFIGS[args.cache]
    print(f"Building cache '{args.cache}' with config: {config}")

    # is_train/train_test_split only affect which cached files this particular instance's
    # file_names ends up listing - _process() itself walks and caches every raw mesh under
    # raw_folder regardless, so a single call processes the entire dataset.
    Teeth3DSDataset("data/3dteethseg", verbose=True, force_process=False,
                     is_train=True, train_test_split=1, **config)

    print(f"Done building cache '{args.cache}'.")
