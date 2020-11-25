import sys
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtWidgets import (QLabel, QFrame, QPushButton, QMainWindow, QWidget,
                             QGridLayout)
from math import sqrt


class GivenTile(QLabel):
    """
    Grid square containing a number given as part of the starting configuration
    of the puzzle - just a label with no methods
    """

    def __init__(self, value, parent=None):
        super().__init__(parent)

        self.setFrameShape(QFrame.StyledPanel)
        self.setAlignment(Qt.AlignCenter)
        self.setText(str(value))


class PlayTile(QPushButton):
    """
    User-editable tile given as a blank tile at the start of the game
    """

    def __init__(self, parent=None, s=50):
        super().__init__(parent)

        size = QSize(s, s)
        self.setMaximumSize(size)
        self.setMinimumSize(size)


class GameWindow(QMainWindow):
    """
    Main window for gameplay
    """

    def __init__(self, puzzle_matrix):
        super().__init__()

        self.puzzle_matrix = puzzle_matrix
        self.setWindowTitle("SUDOKU")

        grid_widget = QWidget()
        # here the width is set to fit nicely around the grid given the default
        # tile size of 50x50
        grid_widget.setFixedSize(480, 480)
        self.setCentralWidget(grid_widget)

        self.puzzle_layout = QGridLayout(grid_widget)
        self.puzzle_layout.setHorizontalSpacing(0)
        self.puzzle_layout.setVerticalSpacing(0)
        self._makePuzzle()

    def _makePuzzle(self):
        squares_per_box = int(sqrt(len(self.puzzle_matrix)))
        gridlayout_size = squares_per_box**2+squares_per_box+1
        tile_iter_x = 0
        for i in range(gridlayout_size):
            tile_iter_y = 0
            for j in range(gridlayout_size-1):
                if i == 0:
                    # top outer border
                    self._addPuzzleGridBorderElement(i, j, QFrame.HLine, inner=False)
                elif i == gridlayout_size-1:
                    # bottom outer border
                    self._addPuzzleGridBorderElement(i, j, QFrame.HLine, inner=False)
                elif i%(squares_per_box+1) == 0:
                    # internal box horizontal border
                    self._addPuzzleGridBorderElement(i, j, QFrame.HLine)
                elif j%(squares_per_box+1) == 0:
                    # internal box vertical border
                    self._addPuzzleGridBorderElement(i, j, QFrame.VLine)
                else:
                    squareval = self.puzzle_matrix[tile_iter_x, tile_iter_y]
                    if squareval > 0:
                        gridsquare = GivenTile(squareval)
                    else:
                        gridsquare = PlayTile()
                    self.puzzle_layout.addWidget(gridsquare, i, j)
                    tile_iter_y += 1
            self._addPuzzleGridBorderElement(i, 0, QFrame.VLine, inner=False) # left outer border
            self._addPuzzleGridBorderElement(i, gridlayout_size, QFrame.VLine, inner=False) # right outer border
            if i != 0 and i%(squares_per_box+1) != 0 and i != gridlayout_size+1:
                tile_iter_x += 1

    def _addPuzzleGridBorderElement(self, pos_x, pos_y, shape, inner=True):
        border_element = QFrame()
        border_element.setFrameShape(shape)
        border_element.setLineWidth(2 if inner else 3)
        self.puzzle_layout.addWidget(border_element, pos_x, pos_y)


# QUICK TEST
if __name__ == '__main__':
    from PyQt5.QtWidgets import QApplication
    from numpy import array
    import sys

    puzzle_array = array([
        [0, 1, 0, 3, 0, 0, 0, 0, 7],
        [4, 0, 9, 0, 0, 0, 0, 0, 2],
        [0, 5, 0, 0, 0, 0, 0, 0, 1],
        [8, 0, 0, 0, 1, 5, 0, 2, 0],
        [0, 0, 0, 4, 0, 2, 6, 9, 0],
        [0, 0, 0, 9, 8, 3, 0, 0, 0],
        [0, 0, 0, 0, 5, 0, 0, 0, 0],
        [0, 0, 0, 7, 4, 0, 0, 0, 5],
        [7, 2, 5, 0, 0, 0, 0, 4, 0]
    ])

    app = QApplication(sys.argv)
    gwin = GameWindow(puzzle_array)
    gwin.show()

    sys.exit(app.exec_())

