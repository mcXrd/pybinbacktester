from typing import List
import numpy as np
import pandas as pd


class Strategy:
    def pick_currency(self, model_output: pd.Series, currencies: List[str]) -> str:
        raise NotImplementedError()

    def pick_side(self, model_output: pd.Series) -> int:
        raise NotImplementedError()

    def do_trade(self, model_output: pd.Series) -> bool:
        raise NotImplementedError()


class MeanStrategy(Strategy):
    def _get_max_index(self, model_output: pd.Series):
        buff = []
        means = []
        for one in model_output:
            if len(buff) < 4:
                buff.append(one)
                if len(buff) == 2:
                    buff.append(one)
            else:
                means.append(np.mean(np.array(buff)))
                buff = []

        abs_means = np.abs(np.array(means))
        max_index = np.argmax(abs_means)
        return max_index

    def pick_currency(self, model_output: pd.Series, currencies: List[str]) -> str:
        max_index = self._get_max_index(model_output)
        return currencies[max_index]

    def pick_true_change_index(self, model_output: pd.Series):
        # order is {0:3h,1:2h,2:1h,.....}
        max_index = self._get_max_index(model_output)
        return max_index * 3 + 1

    def pick_side(self, model_output: pd.Series) -> int:
        true_change_index = self.pick_true_change_index(model_output)
        return 1 if model_output[true_change_index] > 0 else -1

    def do_trade(self, model_output: pd.Series):
        max_index = self._get_max_index(model_output)
        return abs(model_output[max_index]) > 0.01

    def do_trade_explanation(self, model_output: pd.Series):
        max_index = self._get_max_index(model_output)
        return str(abs(model_output[max_index]))

    def do_skip(self) -> bool:
        to_skip = 1  # 0 is for 1h ; 1 is for 2h ; 2 is for 3h
        try:
            getattr(self, "__do_skip_buffer")
        except AttributeError:
            setattr(self, "__do_skip_buffer", to_skip)

        if getattr(self, "__do_skip_buffer") == 0:
            setattr(self, "__do_skip_buffer", to_skip)
            return False
        else:
            setattr(self, "__do_skip_buffer", getattr(self, "__do_skip_buffer") - 1)
            return True


class Direct2hStrategy(MeanStrategy):
    def _get_max_index(self, model_output: pd.Series):
        r = []
        for i in range(int(len(model_output) / 3)):
            index = i * 3 + 0
            r.append(model_output[index])
        abs_val = np.abs(np.array(r))
        max_index = np.argmax(abs_val)
        return max_index

    def pick_true_change_index(self, model_output: pd.Series):
        # order is {0:3h,1:2h,2:1h,.....}
        max_index = self._get_max_index(model_output)
        return max_index * 3 + 0
