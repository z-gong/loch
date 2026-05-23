import click

from .gcmc import gcmc
from .extract import extract


@click.group(context_settings={'show_default': True, 'help_option_names': ['-h', '--help']})
def main():
    """Loch: GPU-accelerated GCMC simulation engine."""
    pass


main.add_command(gcmc)
main.add_command(extract)
