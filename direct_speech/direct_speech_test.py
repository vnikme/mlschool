import direct_speech
import json
import sys


def main():
    text = "Сотрудники ФСБ вызывают на беседы родственников уехавших граждан и уговаривают вернуть родных в Россию. Об этом сообщает правозащитный проект «Первый отдел». Речь идет о людях, которым силовики угрожали делами о госизменах после начала войны; многие из них уехали, опасаясь за свою свободу. «Видимо, кто-то очень хочет повышения по службе за поиск «шпионов». Нам известно минимум о пяти таких случаях», ― отмечают правозащитники. «К кому-то пришли за то, что он делал перевод в украинский фонд, к кому-то ― за то, что он высказыва"
    for quote_open_position, quote_close_position, separator_end_position, trigger_end_position, clarification_begin_position, clarification_end_position in direct_speech.iterate_direct_speech(text):
        if True or '"' in text[quote_open_position + 1 : quote_close_position - 1]:
            print(text[quote_open_position:quote_close_position])
            #print(text[quote_close_position:separator_end_position])
            print(text[separator_end_position:trigger_end_position])
            print(text[clarification_begin_position:clarification_end_position])
            print("")


if __name__ == '__main__':
    main()

