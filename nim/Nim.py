def nim_game(pile, max_sticks):
    print("Current pile: {}".format(pile))
    while pile:
        print("Player 1 turn")
        sticks_taken = int(input("Choose sticks (1-{}): ".format(max_sticks)))
        while sticks_taken > max_sticks or sticks_taken > pile:
            sticks_taken = int(input("Invalid choice, Choose sticks (1-{}): ".format(min(pile, max_sticks))))
        pile -= sticks_taken
        print("Current pile: {}".format(pile))
        if not pile:
            print("Player 1 wins!")
            return
        print("Player 2 turn")
        sticks_taken = int(input("Choose sticks (1-{}): ".format(max_sticks)))
        while sticks_taken > max_sticks or sticks_taken > pile:
            sticks_taken = int(input("Invalid choice, Choose sticks (1-{}): ".format(min(pile, max_sticks))))
        pile -= sticks_taken
        print("Current pile: {}".format(pile))
        if not pile:
            print("Player 2 wins!")
            return

nim_game(20, 4)