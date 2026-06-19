def print_states():
    states = []
    for w in range(3):
        for s in range(3):
            for i in range(4):
                for p in range(2):
                    for sw in range(2):
                        for a in range(2):
                            for b in range(2):
                                for e in range(2):
                                    for g in range(2):
                                        if g > 0 and e == 0: continue
                                        if e > 0 and b == 0: continue
                                        if b > 0 and (sw == 0 or a == 0): continue
                                        states.append((w, s, i, p, sw, a, b, e, g))
                                        if w == 0 and s == 0 and i == 1 and p == 1 and sw == 1 and a == 0 and b == 0 and e == 0 and g == 0:
                                            print("FOUND IT!")

    print(f"Total valid states: {len(states)}")

print_states()
