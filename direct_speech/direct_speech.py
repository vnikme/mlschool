import json
import sys


"""
    null
        « -> open quote {open position, open count}
        -> null
    open quote
        » -> direct speech separator check {open position, close position}
        -> open quote (check quote balance)
    direct speech separator check
        .?! -> null
        non-alphanumeric -> remember separator
        alphanumeric: check separator sequence
            if it's ", -" or so -> direct speech trigger check {open position, close position}
            else null
    direct speech trigger check
        alphanumeric -> remember trigger
        non-alphanumeric: check trigger word
            if in list -> remember clarification {open position, close position, separator seq, trigger}
    remember clarification
        .?!\n -> null
        -> continue
"""

class AbstractState(object):
    def __init__(self, text, objects):
        self.text = text
        self.objects = objects

    def OnSymbol(self, pos):
        return self


class InitialState(AbstractState):
    def __init__(self, text, objects):
        AbstractState.__init__(self, text, objects)

    def OnSymbol(self, pos):
        sym = self.text[pos]
        if sym in '«“':
            return OpenQuoteState(self.text, self.objects, pos)
        if sym in '"':
            return SimpleQuoteState(self.text, self.objects, pos)
        return self


class OpenQuoteState(AbstractState):
    def __init__(self, text, objects, open_position):
        AbstractState.__init__(self, text, objects)
        self.open_position = open_position
        self.quote_balance = 1
        self.is_prev_semicolon = False

    def OnSymbol(self, pos):
        sym = self.text[pos]
        if sym in '«“':
            self.quote_balance += 1
            self.is_prev_semicolon = False
        elif sym in '»”':
            self.quote_balance -= 1
            if self.quote_balance == 0:
                if self.is_prev_semicolon:
                    return DirectSpeechClarificationState(self.text, self.objects, self.open_position, pos + 1, pos + 1, pos + 1)
                else:
                    return DirectSpeechSeparatorState(self.text, self.objects, self.open_position, pos + 1)
            self.is_prev_semicolon = False
        elif sym == ',':
            self.is_prev_semicolon = True
        else:
            self.is_prev_semicolon = False
        return self


class SimpleQuoteState(AbstractState):
    def __init__(self, text, objects, open_position):
        AbstractState.__init__(self, text, objects)
        self.open_position = open_position
        self.quote_balance = 1
        self.is_prev_semicolon = False
        self.is_prev_space = False

    def OnSymbol(self, pos):
        sym = self.text[pos]
        if sym in '"' and self.is_prev_space:
            self.quote_balance += 1
            self.is_prev_space = False
        elif sym in '"' and not self.is_prev_space:
            self.quote_balance -= 1
            self.is_prev_space = False
            if self.quote_balance == 0:
                if self.is_prev_semicolon:
                    return DirectSpeechClarificationState(self.text, self.objects, self.open_position, pos + 1, pos + 1, pos + 1)
                else:
                    return DirectSpeechSeparatorState(self.text, self.objects, self.open_position, pos + 1)
        if sym == ',':
            self.is_prev_semicolon = True
        else:
            self.is_prev_semicolon = False
        if sym == ' ':
            self.is_prev_space = True
        else:
            self.is_prev_space = False
        return self


class DirectSpeechSeparatorState(AbstractState):
    def __init__(self, text, objects, quote_open_position, quote_close_position):
        AbstractState.__init__(self, text, objects)
        self.quote_open_position = quote_open_position
        self.quote_close_position = quote_close_position

    def OnSymbol(self, pos):
        sym = self.text[pos]
        if sym in '.?!\n\0':
            return InitialState(self.text, self.objects)
        if not sym.isalpha() and not sym.isdigit():
            return self
        separator_sequence = self.text[self.quote_close_position:pos]
        for sequence_to_look in [',-', ', -', '.-', '. -', '?-', '? -', '!-', '! -']:
            if sequence_to_look in separator_sequence:
                state = DirectSpeechTriggerState(self.text, self.objects, self.quote_open_position, self.quote_close_position, pos)
                return state.OnSymbol(pos)
        state = InitialState(self.text, self.objects)
        return state.OnSymbol(pos)


class DirectSpeechTriggerState(AbstractState):
    def __init__(self, text, objects, quote_open_position, quote_close_position, separator_end_position):
        AbstractState.__init__(self, text, objects)
        self.quote_open_position = quote_open_position
        self.quote_close_position = quote_close_position
        self.separator_end_position = separator_end_position

    def OnSymbol(self, pos):
        sym = self.text[pos]
        if sym.isalpha() or sym.isdigit():
            return self
        trigger = self.text[self.separator_end_position:pos]
        state = DirectSpeechClarificationState(self.text, self.objects, self.quote_open_position, self.quote_close_position, self.separator_end_position, pos)
        return state.OnSymbol(pos)


class DirectSpeechClarificationState(AbstractState):
    def __init__(self, text, objects, quote_open_position, quote_close_position, separator_end_position, trigger_end_position):
        AbstractState.__init__(self, text, objects)
        self.quote_open_position = quote_open_position
        self.quote_close_position = quote_close_position
        self.separator_end_position = separator_end_position
        self.trigger_end_position = trigger_end_position
        self.clarification_begin_position = None

    def OnSymbol(self, pos):
        sym = self.text[pos]
        if self.clarification_begin_position is None and sym not in ' ':
            self.clarification_begin_position = pos
        if sym in '.?!\n\0':
            self.objects.append((self.quote_open_position, self.quote_close_position, self.separator_end_position, self.trigger_end_position, self.clarification_begin_position, pos))
            return InitialState(self.text, self.objects)
        return self


def iterate_direct_speech(text):
    if '{' in text or '}' in text:
        text = ""
    elif text and text[-1] != '\0':
        text += '\0'
    text = text.replace(chr(160), ' ').replace(chr(10), ' ').replace('—', '-').replace('–', '-').replace('―', '-')
    objects = []
    state = InitialState(text, objects)
    for pos in range(len(text)):
        state = state.OnSymbol(pos)
    for quote_open_position, quote_close_position, separator_end_position, trigger_end_position, clarification_begin_position, clarification_end_position in objects:
        yield quote_open_position, quote_close_position, separator_end_position, trigger_end_position, clarification_begin_position, clarification_end_position


def main():
    data = json.load(open(sys.argv[1], 'rt'))
    direct_speech_count, news_count = 0, 0
    for obj in data:
        for text in obj['text'].split('\n'):
            news_count += 1
            has_direct_speech = False
            for quote_open_position, quote_close_position, separator_end_position, trigger_end_position, clarification_begin_position, clarification_end_position in iterate_direct_speech(text):
                if True or '"' in text[quote_open_position + 1 : quote_close_position - 1]:
                    print(text[quote_open_position:quote_close_position])
                    #print(text[quote_close_position:separator_end_position])
                    #print(text[separator_end_position:trigger_end_position])
                    print(text[clarification_begin_position:clarification_end_position])
                    print("")
                has_direct_speech = True
            if has_direct_speech:
                direct_speech_count += 1
            #if not has_direct_speech:
            #    print('***\n[{}]\n***\n\n'.format(text))
    print(direct_speech_count / news_count, direct_speech_count, news_count)


if __name__ == '__main__':
    main()

