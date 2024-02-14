import tensorflow
from keras.optimizers.schedules.learning_rate_schedule import LearningRateSchedule as LearningRateSchedule


# Learning Rate Scheduler for the optimizer
class LRSchedule(LearningRateSchedule):
    def __init__(self, post_warmup_learning_rate, warmup_steps):
        super().__init__()

        # Tasso di apprendimento dopo il periodo di warmup
        self.post_warmup_learning_rate = post_warmup_learning_rate

        # Numero di passi di warmup
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        # Implementazione della chiamata per calcolare il tasso di apprendimento in base al passo
        global_step = tensorflow.cast(step, tensorflow.float32)
        warmup_steps = tensorflow.cast(self.warmup_steps, tensorflow.float32)

        # Calcolo del progresso di warmup
        warmup_progress = global_step / warmup_steps

        # Calcolo del tasso di apprendimento durante il warmup
        warmup_learning_rate = self.post_warmup_learning_rate * warmup_progress

        # Condizione per determinare se siamo ancora nel periodo di warmup
        return tensorflow.cond(
            global_step < warmup_steps,
            lambda: warmup_learning_rate,
            lambda: self.post_warmup_learning_rate,
        )

    def get_config(self):
        # Funzione per ottenere la configurazione del Learning Rate Scheduler
        return {
            'post_warmup_learning_rate': self.post_warmup_learning_rate,
            'warmup_steps': self.warmup_steps,
        }
