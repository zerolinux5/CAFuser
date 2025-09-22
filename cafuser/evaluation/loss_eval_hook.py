from detectron2.engine.hooks import HookBase
from detectron2.evaluation import inference_context
from detectron2.utils.logger import log_every_n_seconds
from detectron2.data import DatasetMapper, build_detection_test_loader
import detectron2.utils.comm as comm
import torch
import time
import datetime
from torch import nn
from contextlib import ExitStack, contextmanager

class LossEvalHook(HookBase):
    def __init__(self, eval_period, model, data_loader):
        self._model = model
        self._period = eval_period
        self._data_loader = data_loader
    
    def _do_loss_eval(self):
        # Copying inference_on_dataset from evaluator.py
        total = len(self._data_loader)
        num_warmup = min(5, total - 1)
            
        start_time = time.perf_counter()
        total_compute_time = 0
        losses = []
        print("Started")
        with ExitStack() as stack:
            if isinstance(self._model, nn.Module):
                stack.enter_context(inference_context(self._model))
            stack.enter_context(torch.no_grad())
            for idx, inputs in enumerate(self._data_loader):
                if idx == num_warmup:
                    start_time = time.perf_counter()
                    total_compute_time = 0
                start_compute_time = time.perf_counter()
                total_compute_time += time.perf_counter() - start_compute_time
                iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
                seconds_per_img = total_compute_time / iters_after_start
                if idx >= num_warmup * 2 or seconds_per_img > 5:
                    total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                    eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                    log_every_n_seconds(
                        logging.INFO,
                        "Loss on Validation  done {}/{}. {:.4f} s / img. ETA={}".format(
                            idx + 1, total, seconds_per_img, str(eta)
                        ),
                        n=5,
                    )
                print(f"iteration: {idx}")
                print(f"Input: {type(inputs)}")
                print(f"Input len: {len(inputs)}")
                print(f"Input shape: {inputs.shape}")
                output = self._model(inputs)
                print(f"post model: {idx}")
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    print("finished syncing")

                loss_batch = self._get_loss(output)
                print(f"Custom: total loss: {loss_batch}")
                losses.append(loss_batch)
            print("Finished")
            print(losses)
            mean_loss = np.mean(losses)
            print("Finished calculating loss")
            self.trainer.storage.put_scalar('validation_loss', mean_loss)
            comm.synchronize()
            print("Finished synchronizing")

        return losses
            
    def _get_loss(self, metrics_dict):
        print("inside function")
        print(type(metrics_dict[0]["sem_seg"]))
        print(metrics_dict[0]["sem_seg"])
        print(metrics_dict[0]["sem_seg"].shape)
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        print("finished metrics")
        total_losses_reduced = sum(loss for loss in metrics_dict.values())
        return total_losses_reduced
        
        
    def after_step(self):
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if is_final or (self._period > 0 and next_iter % self._period == 0):
            print("Starting info")
            self._do_loss_eval()
        # self.trainer.storage.put_scalars(timetest=12)


@contextmanager
def inference_context(model):
    """
    A context where the model is temporarily changed to eval mode,
    and restored to previous mode afterwards.

    Args:
        model: a torch Module
    """
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)