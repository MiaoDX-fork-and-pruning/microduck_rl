"""List Microduck IsaacLab tasks without initializing Isaac Sim."""

from isaaclab_microduck import available_tasks


def main() -> None:
    for task in available_tasks():
        print(task)


if __name__ == "__main__":
    main()
