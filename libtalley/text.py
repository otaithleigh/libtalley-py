import textwrap
import warnings


class Boxer():
    """Class for creating plaintext "boxes".
    
    A box is defined by several parameters, laid out so::

        |<------------------ width ------------------>|
        <first><------------ rule -------------><right>
        <left><lpad><------- text -------><rpad><right>
        <left><lpad><------- text -------><rpad><right>
        <left><lpad><------- text -------><rpad><right>
        <left><lpad><------- text -------><rpad><right>
        <left><------------- rule -------------><final>
    """

    def __init__(self, left, right, rule, pad=' ', first=None, final=None, rpad=None):
        """Create a new Boxer.

        Parameters
        ----------
        left : str
            The string to use for the left side of the box.
        right : str
            The string to use for the right side of the box.
        rule : str
            The string to use for the rule at the top and bottom of the box.
        pad : str, optional
            The padding to use between `left`/`right` and the text in the box.
            (default: ' ')
        first : str, optional
            Alternate string to use as the first left-hand side, e.g. for multi-
            line comment style boxes. (default: None)
        final : str, optional
            Alternate string to use for the end of the box, e.g. for multiline
            comment style boxes. (default: None)
        rpad : str, optional
            Alternate padding to use for the right-hand side of the box.
            (default: None)
        """
        self.left = left
        self.right = right
        self.rule = rule
        self.first = left if first is None else first
        self.final = right if final is None else final
        self.lpad = pad
        self.rpad = pad if rpad is None else rpad

    def textwidth(self, width=80):
        """Return the available width in characters for the Boxer.
        
        Parameters
        ----------
        width : int, optional
            Total box width to calculate from.
        """
        # textwidth is from the box size, minus the two ends, minus two pads
        return width - len(self.left) - len(self.right) - len(self.lpad) - len(self.rpad)

    def box(self, text, width=80, wrap=True):
        """Box some text, returned as a joined string.

        Parameters
        ----------
        text : str
            Text to box.
        width : int, optional
            Total width of the box. (default: 80)
        wrap : bool, optional
            If True, wrap long lines in `text`. If False, warnings will be issued
            for long lines but they will be left as-is, creating a spiky box.
            (default: True)
        """
        return '\n'.join(self.boxsplit(text, width, wrap))

    def boxsplit(self, text, width=80, wrap=True):
        """Box some text, returned as a list of lines.

        Parameters
        ----------
        text : str
            Text to box.
        width : int, optional
            Total width of the box. (default: 80)
        wrap : bool, optional
            If True, wrap long lines in `text`. If False, warnings will be issued
            for long lines but they will be left as-is, creating a spiky box.
            (default: True)
        """
        textwidth = self.textwidth(width)
        toprulewidth = width - len(self.first) - len(self.right)
        bottomrulewidth = width - len(self.left) - len(self.final)

        # If `rule` is more than one character long, the logic for repeating it
        # is more complicated. We can fit it to the desired width via::
        #
        #   rule*(width // len(rule)) + rule[:width % len(rule)]
        #
        # The first will repeat `rule` until the next repetition would overfill
        # the width; the second takes only enough characters from `rule` to
        # fill out the rest.
        if len(self.rule) == 0:
            toprule = self.first
            bottomrule = self.final
        elif len(self.rule) == 1:
            toprule = self.first + self.rule*toprulewidth + self.right
            bottomrule = self.left + self.rule*bottomrulewidth + self.final
        else:
            toprule = (
                self.first + self.rule*(toprulewidth // len(self.rule)) +
                self.rule[:toprulewidth % len(self.rule)] + self.right
            )
            bottomrule = (
                self.left + self.rule*(bottomrulewidth // len(self.rule)) +
                self.rule[:bottomrulewidth % len(self.rule)] + self.final
            )

        lines = [toprule]
        text = text.splitlines()

        left = self.left + self.lpad
        right = self.rpad + self.right

        for i, line in enumerate(text):
            if wrap and len(line) > textwidth:
                wrappedlines = textwrap.wrap(line, width=textwidth)
            else:
                if len(line) > textwidth:
                    warnings.warn(f"box: line {i} exceeds box dimensions")
                wrappedlines = [line]

            for wline in wrappedlines:
                lines.append(left + wline.ljust(textwidth) + right)

        lines.append(bottomrule)
        return lines


_CBoxer = Boxer(left=' *', right='* ', rule='*', first='/*', final='*/')
_PyBoxer = Boxer(left='#', right='#', rule='=')


def cbox(text, width=80, wrap=True):
    """Place text in a C-style multiline comment box.
    
    Parameters
    ----------
    text : str
        Text to box.
    width : int, optional
        Width of the box. (Note: not text width) (default: 80)
    wrap : bool, optional
        If True, wrap long lines in `text`. If False, warnings will be issued for
        long lines but they will be left as-is, creating a spiky box. (default: True)

    Example
    -------
    >>> print(cbox("hello world!"))
    /******************************************************************************
     * hello world!                                                               *
     ******************************************************************************/
    """
    return _CBoxer.box(text, width=width, wrap=wrap)
