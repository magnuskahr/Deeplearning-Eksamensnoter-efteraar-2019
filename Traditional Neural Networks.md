# 1. Traditional Neural Networks

_Lektion 3 og 4_ 

## 0. Motivation

## 1. Lineær regression

Hvad er regression? Med regression finder man den funktion der bedst beskriver sin data.
Ideen bag *lineær regression*, er så at man finder en lineær funktion (altså en lige linje), der bedst beskriver forholdet mellem to variabler, f.eks detaljerne ved et hus og prisen heraf.
Dette skal forgå ved inputtet $x \in \R^m$ der give outputtet $y \in \R$.

Målet ved at gøre dette, er at vi skal kunne fremsige hvad en resultat $y$ er for en vektor af input værdier $x$.

## 1.1 Eksempel, huse
Lad os beskrive hvad vi kan bruge linear regression med, via et eksempel.

Hvis vi har et trænings-sæt af huse, så definere vi det _i’te_ hus’s detaljer (eller hvad vi kan kalde for _features_) for $x^i$ og prisen er angivet $y^i$.
> *Hus(i)*: Feature: $x^i$, pris: $y^i$ 

Har vi $n$ antal huse, hver med $m$ antal features, kan vores trænings sæt defineres ved:

> $\{y^i, x^i_1, x^i_2, ..., x^i_m\}^n_{i=1}$

Ud fra alt den her data, vil linear regression kunne hjælpe os med at lave en funktion $y=h(x)$, så $y^i\approx h(x^i)$

> $y=h(x)$ hvor $y^i\approx h(x^i)$

Og hvis det lykkedes os med at finde sådan en $h(x)$ funktion, så håber vi at kunne bruge den til at forudsige priser på andre huse.

Ved lineær regression, vil en sådan funktion udtage sig på formen:

> $y = a \cdot x + b$

## 2. Optimisering

For at kunne finde sådan en funktion $y=h(x)$, så vil vi gerne snakke om optimering.

For at optimere en lige linje, vil vi gerne at der er mindst mulig distance mellem linjerne og punkter, og den ved den korteste distance snakker vi da om L2 normen. 


* Explain conceptually what optimization is, and what is the purpose of it

### 2.1 Loss funktioner

Det lader os begynder at snakke om *loss funktioner*, der er forskellige metoder hvorpå vi kan optimere en funktion til at repræsentere et forhold mellem tal.

Hvis vi forestiller os, at vi optimere vores funktion over flere omgange, så for hver omgang, vil vores *loss* funktion fortælle os hvor meget vores funktion afviger fra vores data. Vi ønsker derved at vi har en lille loss, og at vi for hver omgang i vores optimere får en mindre loss, da det fortæller os at vores funktion bliver bedre til at repræsentere vores data.

#### 2.1.1 L2 

Som sagt før, så svarer L2 normen til den korteste distance, givet ved

> $||x||_2 = \sqrt{x_1^2 + x_2^2 + ... + x_n^2}$

At bruge den som _loss_ funktion, så bruges den til at minimere fejlen af summen af alle de kvadrerede forskelle mellem de rigtige værdier og de forudsagte værdier.

> $\sum_{i=1}^n(y_i - \bar{y}_i)^2$

Hvor $\bar{y}_i$ er vores forudsagte value.

Værdien heraf er da vores _loss_, men hvordan bruger vi den så?

### 2.2 Gradient descent


Hvis vi siger at _loss’en_ for vores funktion og dets koefficienter er plottet og den ligner en skål, en tilfældig position på skålens overflade er da _loss’en_ for de koefficienter der er brugt der.

Vi ønsker at finde de koefficenter der er på bunden af skålen, hvor lossen er mindst, kaldet funktionens minimum.

For at komme derned af, bruger vi _gradient descent_, der tager få udregninger fra den optimale løsning hvor vi har en større loss, og efterhånden som vi får en mindre loss, vil der laves flere udregninger.

_Gradient descent_ tager derved store skridt når den er langt væk, og små skridt når den er tæt på.

Matematisk set, differentiere man sin _loss_ funktion og søger den værdi der er mindst. Starter man et tilfældigt sted _x_ og ser hvad kurvens hældning er der, så udregnes der et _stepsize_ ud fra denne, så næste hældning måles ved $x’$. Sådan et stepsize kan man udregne ved $s = y \cdot l$, hvor $l$ er en learning rate, f.eks. $0.1$, hvilket gør at vi hele tiden vil bevæger os mod 0. Man holder så også øje med hældningens fortegn, hvor hvis den ligepludselig ændre sig, har man måske taget et for stort skridt. Dog kan det også være at man blot er stødt på lokal minimum.

Det virker ved, at vi udregner en gradient for vores loss funktion af vores funktion, for vores L2 loss af $h(x)$ vil gradienten bestå af den afledte funktion med respekt til $a$ og respekt til $b$. Vi bruger så denne gradient til at _descende_ ned til det laveste punkt i loss funktionen.

I gradient descent, bruger vi _partial derivative_ til at forstå forskellen i loss for hvad der sker isoleret set når vi ændre en variabel.

~~Når man har to eller flere afledninger af den samme funktion, så kaldes det en _gradient_.~~

Vi kan indsætte vores ligning for den forudsagte værdi:
> f(x, y) = $\sum_{i=1}^n(y_i - (a \cdot x_i + b))^2$

Vi bruger da det _partial derivative_ for de to værdier $a$ og $b$, til at lave en gradient

$$\Big \{\frac{\delta f}{\delta a}, \frac{\delta f}{\delta b}\Big \}$$

- [ ] (lav bedre Eksempel) Define the partial derivate of the L2 loss used in linear regression
- [ ] Know that polynomial fitting can be implemented using linear regression

### 2.3 Learning rate

vi nævnte før _learning rate_ der er et hyperparameter der bestemmer hvor mere vi justere vores koefficienter af hensyn til vores _loss gradient_. Har man en forkert learning rate, kan ens funktion enten overfitte eller underfitte.
Hvis den overfitter, siger man at den har husket data og derved ikke er god til at generaliser ny data, og hvis den er underfittet, vil den ikke kunne repræsentere variationen i datoen.

~~Hvis learning rate er for lav tager der mange epochs at nå minimum, og er den for høj kan den måske ikke nå derned~~

## 3. Logistic regression

16. Explain what logistic regression is (what are the inputs/outputs, and how is mapping between them described?)
17. Mathematically define the model, h(x), used in logistic regression.
18. Know what a sigmoid function is and explain why it can be used to model probabilities for
two classes
19. Define the (cross entropy) loss function typically used in logistic regression
20. Explain how logistic regression works on images, like MNIST
21. Explain what is meant by regularization
22. Explain how weight decay works

### 3.1 Loss functions: softmax
23. Explain conceptually the difference between using L1 vs L2 regularization
24. Define the softmax function and explain how it c be used to make the model predict class
probabilities for more than two classes
25. Explain how the loss function is calculated for softmax regression – and compare with the
loss function for logistic regression
26. Explain how sofmax regression works on images, like MNIST

### 3.2 Entropy
27. Explain what entropy is (conceptually using weather station example is okay)
28. Explain what cross entropy is (conceptually using weather station example is okay)
29. Mention what KL divergence is and what it “measures”
30. Explain how cross entropy relates to logistic regression and softmax regression