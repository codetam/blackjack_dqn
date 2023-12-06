import numpy as np

def get_card_num(card):
    if card > 39:
        return 10
    if card == 0:
        return 0
    num = card % 10
    if num < 1:
        num = 10
    return num

def get_suit(card):
    suit = card % 4
    if suit == 0:
        return '♠'
    elif suit == 1:
        return '♥'
    elif suit == 2:
        return '♦'
    elif suit == 3:
        return '♣'

def calculate_sum(cards):
    card_numbers = np.array([get_card_num(card) for card in cards])
    result = np.sum(card_numbers)
    if 1 in card_numbers and (result + 10) <= 21:
        result = sum(card_numbers) + 10
    if result > 21:
        result = -1
    return result
    
def hit(player_cards, deck, cards_out):
    # first index = 0
    player_index = np.where(player_cards == 0)[0][0]
    # first index != 0
    deck_index = np.where(deck != 0)[0][0]
    player_cards[player_index] = deck[deck_index]
    cards_out[get_card_num(deck[deck_index]) - 1] += 1
    deck[deck_index] = 0
    result = calculate_sum(player_cards)
    return player_cards, deck, cards_out, result

def dealer_turn(dealer_cards, player_cards, deck, cards_out):
    result = 0
    while -1 < result < 17:
        dealer_cards, deck, cards_out, result = hit(dealer_cards,deck,cards_out)
    if result > -1:
        player_sum = calculate_sum(player_cards)
        if result > player_sum:
            result = -1
        elif result == player_sum:
            result = 0
        else:
            result = 1
    else: 
        result = 1
    return dealer_cards, deck, cards_out, result

def print_gamestate(dealer_cards, player_cards):
    print('Dealer has: ', end="")
    for card in dealer_cards:
        if card != 0:
            print(get_card_num(card),get_suit(card), end=" ")
    print("\n sum: ", calculate_sum(dealer_cards))
    print("You have: ", end="")
    for card in player_cards:
        if card != 0:
            print(get_card_num(card),get_suit(card), end=" ")
    print("\n sum: ", calculate_sum(player_cards))

def turn_init(deck, cards_out):
    player_cards = np.zeros((10,), dtype=np.int8)
    dealer_cards = np.zeros((10,), dtype=np.int8)
    player_cards, deck, cards_out, _ = hit(player_cards, deck, cards_out)
    player_cards, deck, cards_out, _ = hit(player_cards, deck, cards_out)
    dealer_cards, deck, cards_out, _ = hit(dealer_cards, deck, cards_out)
    return player_cards, dealer_cards, deck, cards_out