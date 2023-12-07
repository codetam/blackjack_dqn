import numpy as np
from prettytable import PrettyTable 

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
    RED = '\033[31m'
    RESET = '\033[0m'
    suit = card % 4
    if suit == 0:
        return '♠'
    elif suit == 1:
        return RED + '♥' + RESET
    elif suit == 2:
        return RED + '♦' + RESET
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

def print_gamestate(dealer_cards, player_cards, turn, action, done, outcome):
    num_lines = int((55 - len(f"Turn {turn}")) / 2)
    RED = '\033[31m'
    GREEN = '\033[32m'
    RESET = '\033[0m'
    if action == 0:
        print("> Player chose to stand.\n")
    elif action == 1:
        print("> Player chose to hit.\n")
    elif action == 2:
        print("> Player chose to double down.\n")
    print("-" * num_lines + f"Turn {turn}" + "-" * num_lines)
    if done:    
        size = len(f" TURN ENDED || Reward: {outcome}$ ")
        if outcome > 0:
            outcome = GREEN + "Reward: " + str(outcome) + "$" + RESET
        elif outcome == 0:
            outcome = "Reward: " + str(outcome) + "$"
        else:
            outcome = RED + "Reward: " + str(outcome) + "$" + RESET
        result = f"| TURN ENDED || {outcome} |"
        print("+" + "-" * size + "+")
        print(result)
        print("+" + "-" * size + "+")
    print('Dealer Cards')
    for card in dealer_cards:
        if card != 0:
            print(get_card_num(card),get_suit(card), end=" ")
    print("\nSum: ", calculate_sum(dealer_cards))
    print("\nPlayer Cards")
    for card in player_cards:
        if card != 0:
            print(get_card_num(card),get_suit(card), end=" ")
    print("\nSum: ", calculate_sum(player_cards))

def print_table(num_decks, cards_out):
    GREEN = '\033[32m'
    RESET = '\033[0m'
    cards_table = PrettyTable(["Value", "Drawn", "Left", "Percentage drawn", "Next card"]) 
    total_cards_in_deck = 52 * num_decks - sum(cards_out) 

    drawing_percentages = [(4 * num_decks - value) / total_cards_in_deck for value in cards_out]
    drawing_percentages[9] = (16 * num_decks - cards_out[9]) / total_cards_in_deck
    
    
    for i in range(cards_out.size):
        if i == 9:
            total = 16 * num_decks
        else:
            total = 4 * num_decks
        num_cards_left = total - cards_out[i]

        percentage_printed = "{:.2f}".format(drawing_percentages[i])
        if i == np.argmax(drawing_percentages):
            percentage_printed = GREEN + percentage_printed + RESET
        cards_table.add_row([i+1, 
                            cards_out[i], 
                            num_cards_left, 
                            "{:.2f}".format(cards_out[i] * 100 / total) + " %",
                            percentage_printed])
    print(cards_table) 



def turn_init(deck, cards_out):
    player_cards = np.zeros((10,), dtype=np.int8)
    dealer_cards = np.zeros((10,), dtype=np.int8)
    player_cards, deck, cards_out, _ = hit(player_cards, deck, cards_out)
    player_cards, deck, cards_out, _ = hit(player_cards, deck, cards_out)
    dealer_cards, deck, cards_out, _ = hit(dealer_cards, deck, cards_out)
    return player_cards, dealer_cards, deck, cards_out