# -*- coding: utf-8 -*-
"""Equation of state workflow that can use any code plugin implementing the common relax workflow."""
import inspect

from aiida import orm
from aiida.common import exceptions
from aiida.engine import WorkChain, append_, calcfunction
from aiida.plugins import WorkflowFactory

from aiida_common_workflows.workflows.relax.workchain import CommonRelaxWorkChain


def validate_inputs(value, _):
    """Validate the entire input namespace."""
    if 'scale_factors' not in value and ('scale_count' not in value and 'scale_count' not in value):
        return 'neither `scale_factors` nor the pair of `scale_count` and `scale_increment` were defined.'


def validate_sub_process_class(value, _):
    """Validate the sub process class."""
    try:
        process_class = WorkflowFactory(value)
    except exceptions.EntryPointError:
        return f'`{value}` is not a valid or registered workflow entry point.'

    if not inspect.isclass(process_class) or not issubclass(process_class, CommonRelaxWorkChain):
        return f'`{value}` is not a subclass of the `CommonRelaxWorkChain` common workflow.'


def validate_scale_factors(value, _):
    """Validate the `validate_scale_factors` input."""
    if value and len(value) < 3:
        return 'need at least 3 scaling factors.'


def validate_scale_count(value, _):
    """Validate the `scale_count` input."""
    if value is not None and value < 3:
        return 'need at least 3 scaling factors.'


def validate_scale_increment(value, _):
    """Validate the `scale_increment` input."""
    if value is not None and not 0 < value < 1:
        return 'scale increment needs to be between 0 and 1.'


@calcfunction
def scale_structure(structure: orm.StructureData, scale_factor: orm.Float) -> orm.StructureData:
    """Scale the structure with the given scaling factor."""
    ase = structure.get_ase().copy()
    ase.set_cell(ase.get_cell() * float(scale_factor), scale_atoms=True)
    return orm.StructureData(ase=ase)


class EquationOfStateWorkChain(WorkChain):
    """Workflow to compute the equation of state for a given crystal structure."""

    @classmethod
    def define(cls, spec):
        # yapf: disable
        super().define(spec)
        spec.input('structure', valid_type=orm.StructureData, help='The structure at equilibrium volume.')
        spec.input('scale_factors', valid_type=orm.List, required=False, validator=validate_scale_factors,
            help='The list of scale factors at which the volume and total energy of the structure should be computed.')
        spec.input('scale_count', valid_type=orm.Int, default=lambda: orm.Int(7), validator=validate_scale_count,
            help='The number of points to compute for the equation of state.')
        spec.input('scale_increment', valid_type=orm.Float, default=lambda: orm.Float(0.02),
            validator=validate_scale_increment,
            help='The relative difference between consecutive scaling factors.')
        spec.input_namespace('sub_process', dynamic=True)
        spec.input('sub_process_class', non_db=True, validator=validate_sub_process_class)
        spec.inputs.validator = validate_inputs
        spec.outline(
            cls.run_eos,
            cls.inspect_eos,
        )
        spec.output_namespace('structures', valid_type=orm.StructureData,
            help='The relaxed structures at each scaling factor.')
        spec.output_namespace('total_energies', valid_type=orm.Float,
            help='The computed total energy of the relaxed structure at each scaling factor.')
        spec.exit_code(400, 'ERROR_SUB_PROCESS_FAILED',
            message='At least one of the `{cls}` sub processes did not finish successfully.')

    def run_eos(self):
        """Run the sub process at each scale factor to compute the structure volume and total energy."""
        process_class = WorkflowFactory(self.inputs.sub_process_class)
        process_class_name = process_class.__name__

        if 'scale_factors' in self.inputs:
            scale_factors = self.inputs.scale_factors
        else:
            count = self.inputs.scale_count.value
            increment = self.inputs.scale_increment.value
            scale_factors = [orm.Float(1 + i * increment - (count - 1) * increment / 2) for i in range(count)]

        for scale_factor in scale_factors:
            structure = scale_structure(self.inputs.structure, scale_factor)
            self.report(f'submitting `{process_class_name}` for scale_factor `{scale_factor}`')
            node = self.submit(process_class, structure=structure, **self.inputs.sub_process)
            self.to_context(children=append_(node))

    def inspect_eos(self):
        """Inspect all children workflows to make sure they finished successfully."""
        if any([not child.is_finished_ok for child in self.ctx.children]):
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED.format(cls=self.inputs.sub_process_class)

        for index, child in enumerate(self.ctx.children):
            volume = child.outputs.relaxed_structure.get_cell_volume()
            energy = child.outputs.total_energy.value
            self.report(f'Image {index}: volume={volume}, total energy={energy}')
            self.out(f'structures.{index}', child.outputs.relaxed_structure)
            self.out(f'total_energies.{index}', child.outputs.total_energy)
