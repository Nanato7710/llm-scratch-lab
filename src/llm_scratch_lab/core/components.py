from llm_scratch_lab.core.registry import ComponentRegistry


def create_default_registry() -> ComponentRegistry:
    from llm_scratch_lab.data.hf_streaming import register as register_data
    from llm_scratch_lab.methods.causal_pretraining import register as register_method
    from llm_scratch_lab.models.gemma3 import register as register_model
    from llm_scratch_lab.optimizers import register as register_optimizer

    registry = ComponentRegistry()
    register_model(registry)
    register_data(registry)
    register_method(registry)
    register_optimizer(registry)
    return registry
