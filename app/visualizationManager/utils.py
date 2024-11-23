import io

from matplotlib import pyplot as plt


def save_figure_to_buffer(format_type: str) -> io.BytesIO:
    buffer = io.BytesIO()
    try:
        plt.savefig(buffer, format=format_type, bbox_inches="tight")
        buffer.seek(0)
    finally:
        plt.close()
    return buffer