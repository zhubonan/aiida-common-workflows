# -*- coding: utf-8 -*-
"""Commands to launch common workflows."""
import functools

import click

from aiida.cmdline.params import options as options_core
from aiida.cmdline.params import types

from aiida_common_workflows.plugins import get_workflow_entry_point_names, load_workflow_entry_point
from .root import cmd_root
from . import options
from . import utils


@cmd_root.group('launch')
def cmd_launch():
    """Launch a common workflow."""


@cmd_launch.command('relax')
@click.argument('plugin', type=types.LazyChoice(functools.partial(get_workflow_entry_point_names, 'relax', True)))
@options.STRUCTURE(help='The structure to relax.')
@options.PROTOCOL(type=click.Choice(['fast', 'moderate', 'precise']), default='fast')
@options.RELAXATION_TYPE()
@options.THRESHOLD_FORCES()
@options.THRESHOLD_STRESS()
@options.DAEMON()
@click.option('--show-engines', is_flag=True, help='Show information on the required calculation engines.')
def cmd_relax(plugin, structure, protocol, relaxation_type, threshold_forces, threshold_stress, daemon, show_engines):
    """Relax a crystal structure using the common relax workflow for one of the existing plugin implementations.

    The command will automatically try to find and load the codes that are required by the plugin workflow. If no code
    is installed for at least one of the calculation engines, the command will fail. Use the `--show-engine` flag to
    display the required calculation engines for the selected plugin workflow.
    """
    # pylint: disable=too-many-locals
    from aiida.orm import QueryBuilder, Code

    process_class = load_workflow_entry_point('relax', plugin)
    generator = process_class.get_inputs_generator()

    if not generator.is_valid_protocol(protocol):
        protocols = generator.get_protocol_names()
        process_class_name = process_class.__name__
        message = f'`{protocol}` is not implemented by `{process_class_name}` workflow: choose one of {protocols}'
        raise click.BadParameter(message, param_hint='protocol')

    if show_engines:
        for engine in generator.get_calc_types():
            schema = generator.get_calc_type_schema(engine)
            click.secho(engine, fg='red', bold=True)
            click.echo('Required code plugin: {}'.format(schema['code_plugin']))
            click.echo('Engine description:   {}'.format(schema['description']))

        return

    engines = {}

    for engine in generator.get_calc_types():
        schema = generator.get_calc_type_schema(engine)
        engines[engine] = {
            'options': {
                'resources': {
                    'num_machines': 1
                },
                'max_wallclock_seconds': 86400,
            }
        }
        code_plugin = schema['code_plugin']
        builder = QueryBuilder().append(Code, filters={'attributes.input_plugin': code_plugin})

        code = builder.first()

        if code is None:
            raise click.UsageError(f'could not find a configured code for the plugin `{code_plugin}`.')

        engines[engine]['code'] = code[0].label

    builder = generator.get_builder(structure, engines, protocol, relaxation_type, threshold_forces, threshold_stress)
    utils.launch_process(builder, daemon)


@cmd_launch.command('eos')
@click.argument('plugin', type=types.LazyChoice(functools.partial(get_workflow_entry_point_names, 'relax', True)))
@options.STRUCTURE(help='The structure to relax.')
@options.PROTOCOL(type=click.Choice(['fast', 'moderate', 'precise']), default='fast')
@options.RELAXATION_TYPE()
@options.THRESHOLD_FORCES()
@options.THRESHOLD_STRESS()
@options.DAEMON()
@click.option('--show-engines', is_flag=True, help='Show information on the required calculation engines.')
def cmd_eos(plugin, structure, protocol, relaxation_type, threshold_forces, threshold_stress, daemon, show_engines):
    """Relax a crystal structure using the common relax workflow for one of the existing plugin implementations.

    The command will automatically try to find and load the codes that are required by the plugin workflow. If no code
    is installed for at least one of the calculation engines, the command will fail. Use the `--show-engine` flag to
    display the required calculation engines for the selected plugin workflow.
    """
    # pylint: disable=too-many-locals
    from aiida.orm import QueryBuilder, Code
    from aiida_common_workflows.plugins import get_entry_point_name_from_class
    from aiida_common_workflows.workflows.eos import EquationOfStateWorkChain

    process_class = load_workflow_entry_point('relax', plugin)
    generator = process_class.get_inputs_generator()

    if not generator.is_valid_protocol(protocol):
        protocols = generator.get_protocol_names()
        process_class_name = process_class.__name__
        message = f'`{protocol}` is not implemented by `{process_class_name}` workflow: choose one of {protocols}'
        raise click.BadParameter(message, param_hint='protocol')

    if show_engines:
        for engine in generator.get_calc_types():
            schema = generator.get_calc_type_schema(engine)
            click.secho(engine, fg='red', bold=True)
            click.echo('Required code plugin: {}'.format(schema['code_plugin']))
            click.echo('Engine description:   {}'.format(schema['description']))

        return

    engines = {}

    for engine in generator.get_calc_types():
        schema = generator.get_calc_type_schema(engine)
        engines[engine] = {
            'options': {
                'resources': {
                    'num_machines': 1
                },
                'max_wallclock_seconds': 86400,
            }
        }
        code_plugin = schema['code_plugin']
        builder = QueryBuilder().append(Code, filters={'attributes.input_plugin': code_plugin})

        code = builder.first()

        if code is None:
            raise click.UsageError(f'could not find a configured code for the plugin `{code_plugin}`.')

        engines[engine]['code'] = code[0]

    builder = generator.get_builder(structure, engines, protocol, relaxation_type, threshold_forces, threshold_stress)
    inputs = {
        'structure': structure,
        'sub_process_class': get_entry_point_name_from_class(process_class).name,
        'sub_process': {
            **builder._inputs(prune=True)  # pylint: disable=protected-access
        }
    }
    inputs['sub_process'].pop('structure', None)
    utils.launch_process(EquationOfStateWorkChain, daemon, **inputs)


@cmd_launch.command('plot-eos')
@options_core.NODE()
def cmd_plot_eos(node):
    """Plot the results from an `EquationOfStateWorkChain`."""
    # pylint: disable=too-many-locals,invalid-name
    import numpy
    import pylab as plt

    from aiida.common import LinkType

    def birch_murnaghan(V, E0, V0, B0, B01):
        """Compute energy by Birch Murnaghan formula."""
        r = (V0 / V)**(2. / 3.)
        return E0 + 9. / 16. * B0 * V0 * (r - 1.)**2 * (2. + (B01 - 4.) * (r - 1.))

    def fit_birch_murnaghan_params(volumes, energies):
        """Fit Birch Murnaghan parameters."""
        from scipy.optimize import curve_fit

        params, covariance = curve_fit(  # pylint: disable=unbalanced-tuple-unpacking
            birch_murnaghan,
            xdata=volumes,
            ydata=energies,
            p0=(
                energies.min(),  # E0
                volumes.mean(),  # V0
                0.1,  # B0
                3.,  # B01
            ),
            sigma=None
        )
        return params, covariance

    outputs = node.get_outgoing(link_type=LinkType.RETURN).nested()

    volumes = []
    energies = []
    for index, structure in sorted(outputs['structures'].items()):
        volumes.append(structure.get_cell_volume())
        energies.append(outputs['total_energies'][index].value)

    params, _ = fit_birch_murnaghan_params(numpy.array(volumes), numpy.array(energies))

    vmin = volumes.min()
    vmax = volumes.max()
    vrange = numpy.linspace(vmin, vmax, 300)

    plt.plot(volumes, energies, 'o')
    plt.plot(vrange, birch_murnaghan(vrange, *params))

    plt.xlabel('Volume (A^3)')
    plt.ylabel('Energy (eV)')
    plt.show()
