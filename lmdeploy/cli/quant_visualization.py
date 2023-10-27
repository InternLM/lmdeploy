# Copyright (c) OpenMMLab. All rights reserved.


class SubCliDraw(object):
    """CLI for compressing LLMs."""

    def draw(self,
             modes=None,
             linear_name=None,
             pretrained_model_name_or_path=None,
             work_dir='work_dir',
             use_input=True,
             key='absmax',
             layers=None,
             force_calibrate=False):
        """This function is used to visualize data from a pre-trained
        transformer model. Depending on the mode, it will leverage different
        visualization functions (draw1, draw2, draw3, draw4).

        Args:
            mode (list of int, optional): Modes determine which visualization
                function(s) will be used. If not provided, all four
                visualization functions are executed.
                Available modes: [1, 2, 3, 4].
            linear_name (str, optional): Name of the linear layer for which
                the visualization will be created. If not provided, it will
                process all available layers.
            pretrained_model_name_or_path (str, optional): Path or name of the
                pretrained model. Required if `force_calibrate` is True or
                no calibration data exists in `work_dir`. Defaults to None.
            work_dir (str, optional): Working directory where intermediate
                files are saved. Defaults to 'work_dir'.
            use_input (bool, optional): Use the input or output of a linear
                layer. Defaults to use the input of a linear layer.
            key (str, optional): The specific feature to plot. Can be 'absmax',
                'absmean', 'max', 'mean' or 'min'. Defaults to 'absmax'.
            layers (list, optional): List of layers to draw. If None,
                all layers will be drawn. Defaults to None.
            force_calibrate (bool, optional): Whether to force recalibration
                even if calibration data exists. Defaults to False.
        """
        from lmdeploy.lite.apis.quant_visualization import draw
        draw(modes, linear_name, pretrained_model_name_or_path, work_dir,
             use_input, key, layers, force_calibrate)
