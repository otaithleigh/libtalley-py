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
    def __init__(self,
                 left,
                 right,
                 rule,
                 pad=' ',
                 first=None,
                 final=None,
                 rpad=None,
                 width=80):
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
        width : int, optional
            Default width in characters of created boxes. (default: 80)
        """
        self.left = left
        self.right = right
        self.rule = rule
        self.first = left if first is None else first
        self.final = right if final is None else final
        self.lpad = pad
        self.rpad = pad if rpad is None else rpad
        self.width = width

    def textwidth(self, width=None):
        """Return the available width in characters for the Boxer.

        Parameters
        ----------
        width : int, optional
            Total box width to calculate from.
        """
        if width is None:
            width = self.width
        # textwidth is from the box size, minus the two ends, minus two pads
        return (width - len(self.left) - len(self.right) - len(self.lpad) -
                len(self.rpad))

    def box(self, text, width=None, wrap=True, strip=True):
        """Box some text, returned as a joined string.

        Parameters
        ----------
        text : str
            Text to box.
        width : int, optional
            Total width of the box. (default: self.width)
        wrap : bool, optional
            If True, wrap long lines in `text`. If False, warnings will be
            issued for long lines but they will be left as-is, creating a spiky
            box. (default: True)
        strip : bool, optional
            If True, strip trailing whitespace from each line. (default: True)
        """
        return '\n'.join(self.boxsplit(text, width, wrap, strip))

    def boxsplit(self, text, width=None, wrap=True, strip=True):
        """Box some text, returned as a list of lines.

        Parameters
        ----------
        text : str
            Text to box.
        width : int, optional
            Total width of the box. (default: self.width)
        wrap : bool, optional
            If True, wrap long lines in `text`. If False, warnings will be
            issued for long lines but they will be left as-is, creating a spiky
            box. (default: True)
        strip : bool, optional
            If True, strip trailing whitespace from each line. (default: True)
        """
        if width is None:
            width = self.width

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
            # yapf: disable
            toprule = (
                self.first + self.rule*(toprulewidth // len(self.rule)) +
                self.rule[:toprulewidth % len(self.rule)] + self.right
            )
            bottomrule = (
                self.left + self.rule*(bottomrulewidth // len(self.rule)) +
                self.rule[:bottomrulewidth % len(self.rule)] + self.final
            )
            # yapf: enable

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
        if strip:
            lines = [l.rstrip() for l in lines]
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
        If True, wrap long lines in `text`. If False, warnings will be issued
        for long lines but they will be left as-is, creating a spiky box.
        (default: True)

    Example
    -------
    >>> print(cbox("hello world!", width=40))
    /**************************************
     * hello world!                       *
     **************************************/
    """
    return _CBoxer.box(text, width=width, wrap=wrap)


def pybox(text, width=80, wrap=True):
    """
    Parameters
    ----------
    text : str
        Text to box.
    width : int, optional
        Width of the box. (Note: not text width) (default: 80)
    wrap : bool, optional
        If True, wrap long lines in `text`. If False, warnings will be issued
        for long lines but they will be left as-is, creating a spiky box.
        (default: True)

    Example
    -------
    >>> print(pybox("hello world!", width=40))
    #======================================#
    # hello world!                         #
    #======================================#
    """
    return _PyBoxer.box(text, width=width, wrap=wrap)
