# -*- coding: utf-8 -*-
"""
All character set and unicode related tests.
"""
from jedi import Script
from jedi._compatibility import u, unicode


def test_unicode_script():
    """ normally no unicode objects are being used. (<=2.7) """
    s = unicode("import datetime; datetime.timedelta")
    completions = Script(s).completions()
    assert len(completions)
    assert type(completions[0].description) is unicode

    s = u("author='öä'; author")
    completions = Script(s).completions()
    x = completions[0].description
    assert type(x) is unicode

    s = u("#-*- coding: iso-8859-1 -*-\nauthor='öä'; author")
    s = s.encode('latin-1')
    completions = Script(s).completions()
    assert type(completions[0].description) is unicode


def test_unicode_attribute():
    """ github jedi-vim issue #94 """
    s1 = u('#-*- coding: utf-8 -*-\nclass Person():\n'
           '    name = "e"\n\nPerson().name.')
    completions1 = Script(s1).completions()
    assert 'strip' in [c.name for c in completions1]
    s2 = u('#-*- coding: utf-8 -*-\nclass Person():\n'
           '    name = "é"\n\nPerson().name.')
    completions2 = Script(s2).completions()
    assert 'strip' in [c.name for c in completions2]


def test_multibyte_script():
    """ `jedi.Script` must accept multi-byte string source. """
    try:
        code = u("import datetime; datetime.d")
        comment = u("# multi-byte comment あいうえおä")
        s = (u('%s\n%s') % (code, comment)).encode('utf-8')
    except NameError:
        pass  # python 3 has no unicode method
    else:
        assert len(Script(s, 1, len(code)).completions())