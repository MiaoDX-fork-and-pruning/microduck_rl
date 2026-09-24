"""IsaacLab actuators and simulator-independent BAM math.

The public actuator classes are lazy so CPU-side parity tools can import
``isaaclab_microduck.actuators.bam_math`` without an Isaac Sim installation.
"""

__all__ = ["BamActuator", "BamActuatorCfg", "LaggedExternalEffort"]


def __getattr__(name: str):
    if name in __all__:
        if name == "LaggedExternalEffort":
            from .physx_friction_bridge import LaggedExternalEffort

            return LaggedExternalEffort
        from .bam_actuator import BamActuator, BamActuatorCfg

        return {"BamActuator": BamActuator, "BamActuatorCfg": BamActuatorCfg}[name]
    raise AttributeError(name)
