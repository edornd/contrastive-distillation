from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from matplotlib.figure import Figure
from saticl.logging import BaseLogger
from saticl.utils.decorators import only_rank


class TensorBoardLogger(BaseLogger):

    def __init__(self,
                 log_folder: Path = Path("logs"),
                 filename_suffix: str = "",
                 current_step: int = 0,
                 icl_step: int = 0,
                 comment: str = "") -> None:
        super().__init__()
        self.log = SummaryWriter(log_dir=log_folder, filename_suffix=filename_suffix, comment=comment)
        self.current_step = current_step
        self.icl_step = icl_step

    def step(self, iteration: int = None) -> None:
        if not iteration:
            self.current_step += 1
        else:
            self.current_step = iteration

    def get_step(self, kwargs: dict) -> int:
        return kwargs.pop("step", self.current_step)

    def transform_name(self, name: str) -> str:
        if self.icl_step is not None:
            name = f"{name}/{self.icl_step}"
        return name

    @only_rank(0)
    def log_model(self, model: nn.Module, input_size: tuple = (1, 4, 256, 256), device: str = "cpu") -> None:
        sample_input = torch.rand(input_size, device=device)
        self.log.add_graph(model, input_to_model=sample_input)

    @only_rank(0)
    def log_scalar(self, name: str, value: float, **kwargs) -> None:
        self.log.add_scalar(self.transform_name(name), value, global_step=self.get_step(kwargs), **kwargs)

    @only_rank(0)
    def log_image(self, name: str, image: np.ndarray, **kwargs) -> None:
        self.log.add_image(self.transform_name(name), image, global_step=self.get_step(kwargs), **kwargs)

    @only_rank(0)
    def log_figure(self, name: str, figure: Figure, **kwargs) -> None:
        self.log.add_figure(self.transform_name(name), figure, global_step=self.get_step(kwargs), **kwargs)

    @only_rank(0)
    def log_table(self, name: str, table: Dict[str, str], **kwargs: dict):
        name = self.transform_name(name)
        table_html = "<table width=\"100%\"> "
        table_html += "<tr><th>Key</th><th>Value</th></tr>"
        # iterate dictionary rows
        for k, v in table.items():
            table_html += f"<tr><td>{k}</td><td>{v}</td></tr>"
        table_html += "</table>"
        # log the table as html text
        self.log.add_text(name, table_html, global_step=self.get_step(kwargs))

    @only_rank(0)
    def log_results(self, name: str, headers: List[str], results: Dict[str, List[float]], **kwargs: dict):
        header_html = "".join([f"<th>{h}</th>" for h in headers])
        table_html = f"<table width=\"100%\"><tr><th>metric/class</th>{header_html}</tr>"
        # iterate results and write them into a table
        for score_name, scores in results.items():
            row_html = "".join([f"<td>{x:.4f}</td>" for x in scores])
            table_html += f"<tr><td>{score_name}</td>{row_html}</tr>"
        table_html += "</table>"
        self.log.add_text(self.transform_name(name), table_html, global_step=self.get_step(kwargs))
