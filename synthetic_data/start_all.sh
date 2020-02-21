# Generates all the data presented in the paper
for a in noisy_1 noisy_2 separ_1 separ_2
do
    # Time to run the following command:
    # for a *_1 type exp: ~ 8-9 min.
    # for a *_2 type exe: ~ 15 min.
    python main.py $a &
done
wait
python main.py boxplots
python main.py dists
python main.py mammen 0.80
python main.py mammen 0.20
python main.py invs
