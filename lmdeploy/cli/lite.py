# Copyright (c) OpenMMLab. All rights reserved.


class SubCliLite(object):
    """CLI for compressing LLMs."""

    def auto_awq(self,
                 model: str,
                 work_dir: str,
                 w_bits: int = 4,
                 w_sym: bool = False,
                 w_group_size: int = 128,
                 device: str = 'cuda'):
        """Perform weight quantization using AWQ algorithm.

        Args:
            model (str): The path of model in hf format.
            work_dir (str): The working directory to save results.
            w_bits (int): Bit number for weight quantization.
            w_sym (bool): Whether to do symmetric quantization.
            w_group_size (int): Group size for weight quantization statistics.
            device (str): Device type of running.
        """
        from lmdeploy.lite.apis.auto_awq import auto_awq

        auto_awq(model,
                 work_dir,
                 w_bits=w_bits,
                 w_sym=w_sym,
                 w_group_size=w_group_size,
                 device=device)

    def calibrate(self,
                  model: str,
                  calib_dataset: str = 'c4',
                  calib_samples: int = 128,
                  calib_seqlen: int = 2048,
                  work_dir: str = './work_dir',
                  device: str = 'cuda') -> None:
        """Perform calibration on a given dataset.

        Args:
            model (str): The model to be loaded.
            calib_dataset (str, optional): The calibration dataset name.
                Defaults to 'c4'.
            calib_samples (int, optional): The number of samples for
                calibration. Defaults to 128.
            calib_seqlen (int, optional): The sequence length for calibration.
                Defaults to 2048.
            work_dir (str): The working directory for outputs.
                Defaults to './work_dir'.
            device (str, optional): The device to be used for calculation.
                Defaults to 'cuda'.
        """
        from lmdeploy.lite.apis.calibrate import calibrate

        calibrate(model,
                  calib_dataset=calib_dataset,
                  calib_samples=calib_samples,
                  calib_seqlen=calib_seqlen,
                  work_dir=work_dir,
                  device=device)

    def kv_qparams(self,
                   work_dir: str,
                   turbomind_dir: str,
                   kv_bits: int = 8,
                   kv_sym: bool = False,
                   num_tp: int = 1) -> None:
        """Export key and value stats.

        Args:
            work_dir (str): Directory path where the stats
                are saved.
            turbomind_dir (str): Directory path where to
                save the results.
            kv_bits (int, optional): Number of bits for quantization.
                Defaults to 8.
            kv_sym (bool, optional): Whether to use symmetric quantization.
                Defaults to False.
            num_tp (int, optional): Number of tensor parallelism.
                Defaults to 1.
        """
        from lmdeploy.lite.apis.kv_qparams import main as run_kv_qparams

        run_kv_qparams(work_dir,
                       turbomind_dir,
                       kv_bits=kv_bits,
                       kv_sym=kv_sym,
                       num_tp=num_tp)

    def get_small_sharded_hf(self, src_dir: str, dst_dir: str):
        """Convert a hugging face model to the smallest sharded one.

        Args:
            src_dir (str): The directory of the input HF model.
            dst_dir (str): The directory to save new  model.
        """
        from lmdeploy.lite.apis.get_small_sharded_hf import main as run_sharded
        run_sharded(src_dir, dst_dir)
