# 1. Dataset and Preprocessing

Dodałem filtr dwustronny, czyli postawiłem zastosować technikę wygładzania z zachowaniem krawędzi. Nie jest to rozwiązanie typu one-size-fits, można zastosować połączenie dwóch technik. Początkowo zastosować jeden ze specjalistycznych filtrów szumu plamkowego następnie technikę zachowania krawędzi np. filtr dwustronny.

Mając więcej czasu, przygotowałbym takie rozwiązanie i porównał działanie.

Ponadto, robiąc research do tego zadania nadtrawiłem na dokument [Speckle Noise Reduction in Ultrasound Images using Denoising Auto-encoder with Skip connection](https://arxiv.org/html/2403.02750v1). Auto enkodery z połączeniami pomijającymi (często określane jako architektury podobne do U-Net) oferują znaczące zalety w zakresie usuwania szumów z obrazów ultradźwiękowych, szczególnie w przypadku wdrażania urządzeń brzegowych.

**Wydajności:**

- Zmniejszona liczba parametrów: Połączenia pomijane umożliwiają płytsze sieci przy zachowaniu wydajności, zmniejszając ogólny rozmiar modelu.
- Szybsza konwergencja: Sieci z połączeniami pomijanymi zazwyczaj wymagają mniejszej liczby iteracji szkoleniowych.
- Mniejsze opóźnienie wnioskowania: Zoptymalizowana architektura umożliwia szybsze czasy przetwarzania — krytyczne dla aplikacji do obrazowania medycznego w czasie rzeczywistym.

**Dokładność:**

- Zachowane szczegóły: Połączenia pomijane pomagają zachować informacje o wysokiej częstotliwości (krawędzie, tekstury), które w przeciwnym razie zostałyby utracone w procesie kodowania.
- Lepsze radzenie sobie z szumem plamkowym: Badania pokazują, że połączenia pomijane są szczególnie skuteczne w przypadku charakterystycznego szumu plamkowego w obrazach ultrasonograficznych.

**Urządzenia brzegowe:**

- Wydajność pamięci: Połączenia pomijane zmniejszają potrzebę dużych map cech pośrednich, obniżając wymagania dotyczące pamięci RAM.
- Potencjał paralelizacji: Architektura dobrze nadaje się do sprzętowe przyspieszenie na specjalnych chipach dedykowanych przyśpieszeniu AI (Neural processing units (NPU)).
- Skalowalność: Liczbę połączeń pomijających można dostosować w zależności od możliwości urządzenia, co zapewnia dobrą równowagę między rozmiarem modelu a wydajnością.

Połączenia pomijające zasadniczo tworzą skróty w sieci, które umożliwiają łatwiejszy przepływ informacji gradientowych podczas treningu i zachowują informacje przestrzenne, które w przeciwnym razie zostałyby utracone w procesie kodowania-dekodowania. Jest to szczególnie cenne w przypadku obrazów ultrasonograficznych, w których zachowanie drobnych szczegółów anatomicznych przy jednoczesnym usuwaniu szumów jest niezbędne.

### Porównanie z klasycznymi metodami:
W porównaniu ze specjalistycznymi filtrami szumów pasmowych (Lee, Frost, Kuan)

**Zalety auto enkoderów:**

- Doskonałe zachowanie krawędzi: Połączenia pomijające zachowują szczegóły strukturalne, które filtry statystyczne często rozmywają.
- Lepsza adaptowalność: Możliwość uczenia się złożonych wzorców szumów specyficznych dla konkretnego sprzętu ultradźwiękowego.

**Wady:**

- Wymagania szkoleniowe: Potrzeba reprezentatywnych danych w przeciwieństwie do filtrów statystycznych, które działają „od razu”
- Bardziej intensywne obliczeniowo: Chociaż tu można na pewno zastosować jakieś techniki optymalizacji (kwantyzacja).

### W porównaniu z adaptacyjnym wyrównaniem histogramu:

**Zalety autoenkoderów:**

- Rzeczywiste usuwanie szumów, a nie tylko poprawa kontrastu.
- Zachowanie struktury: CLAHE może wzmacniać szumy wraz z użytecznymi cechy.
- Uzupełniające podejścia: Auto enkodery działają dobrze, gdy są stosowane z CLAHE.

**Wady:**

- Czas przetwarzania: CLAHE jest niezwykle szybki w porównaniu do rozwiązania z secią neuronową
- Prostota: CLAHE nie wymaga szkolenia

### W porównaniu do wygładzania z zachowaniem krawędzi:

**Zalety auto enkoderów:**

- Przetwarzanie uwzględniające treść: Uczy się, co stanowi szum, a co ważne cechy diagnostyczne.
- Hierarchiczne zachowanie cech: Połączenia pomijające zachowują szczegóły w wielu skalach.
- Lepsza wydajność w przypadku złożonych struktur anatomicznych.

**Wady:**

- Natura czarnej skrzynki: Mniej interpretowane niż techniki takie jak filtrowanie dwustronne
- Złożoność implementacji: Większe wyzwanie w implementacji i optymalizacji

Ocena ogólna
Autoenkodery z połączeniami pomijającymi zazwyczaj zapewniają lepszą jakość usuwania szumów, ale z większym narzutem obliczeniowym.

Najlepsze podejście często łączy metody — na przykład użycie autoenkodera do usuwania szumów, a następnie CLAHE do poprawy kontrastu.

**Dlaczego wybrałem filtrowanie dwustronne?**

1. Bardzo dobre zachowanie krawędzi
- Zachowuje granice anatomiczne, które są kluczowe dla diagnozy
- Zachowuje małe struktury, które inne filtry mogą eliminować
- Nieliniowa natura zachowuje ostre przejścia, wygładzając jednocześnie jednorodne obszary
1. Rozsądne wymagania obliczeniowe
- Można je optymalizować za pomocą technik aproksymacji dla urządzeń brzegowych
- Biblioteki oferują zoptymalizowane implementacji np. OpenCV
1. Brak konieczności szkolenia
- Działa „od razu” bez zbierania danych do treningu
- Parametry można dostroić na podstawie charakterystyki szumu
1. Szczególne zalety dla obrazów ultrasonograficznych
- Radzi sobie z szumem plamkowym lepiej niż filtrowanie Gaussa
- Zachowuje informacje o teksturze lepiej niż filtrowanie medianowe
- Zachowuje lepsze zróżnicowanie tkanek niż filtry Lee lub Frost

Znalazłem informację, że użycie filtru dwustronnego daje 70-80% tego co dałoby zastosowanie auto enkodera a jednocześnie wymaga znaczniej mniej zasobów obliczeniowych. 

## Testy (na podstawie 01_preprocessing_compare.ipynb)

Próbka 1:

- Filtrowanie Gaussa wykazuje **22,8% większą siłę krawędzi** (0,479 w porównaniu do 0,390)
- Wartości pikseli różnią się średnio o 0,02387 (około 2,4% pełnego zakresu intensywności)
- Obie metody dają podobną ogólną jasność (średnia 0,436 w porównaniu do 0,433)

Próbka 2:

- Jeszcze większa różnica w sile krawędzi - **47,3% wyższa w przypadku Gaussa** (0,386 w porównaniu do 0,262)
- Różnice w pikselach wynoszą średnio 0,03118 (około 3,1% pełnego zakresu intensywności)
- Ponownie, podobne poziomy jasności (0,378 w porównaniu do 0,373)

W obu próbkach filtrowanie dwustronne jest wyraźnie bardziej agresywne w redukcji tego, co uważa za szum, jednocześnie będąc bardziej selektywnym w kwestii tego, które krawędzie zachowuje. Istotna różnica w metrykach siły krawędzi (szczególnie w Próbce 2) wskazuje, że filtrowanie dwustronne podejmuje bardziej niuansowe decyzje dotyczące tego, co stanowi znaczącą krawędź.

W przypadku segmentacji ultrasonograficznej w medycynie ta selektywność filtrowania dwustronnego jest cenna, ponieważ:

1. Redukuje szum plamkowy, który mógłby tworzyć fałszywe granice
2. Zachowuje silniejsze krawędzie anatomiczne, które reprezentują rzeczywiste przejścia tkankowe
3. Spójność między próbkami sugeruje, że to zachowanie jest niezawodne

![sample1](images/preprocessing-sample1.png)
![sample2](images/preprocessing-sample2.png)

# 2. U-Net Implementation

Rozważałem dodanie mechanizmu uwagi, który:

- Dodaje wyraźne skupienie się na istotnych cechach obrazu.
- Szczególnie skuteczny w przypadku zadań, których pewne obszary wymagają szczególnej uwagi.
- Pomaga w przypadku drobnych szczegółów i złożonych granic.
- Zwykle wymaga większych zasobów obliczeniowych.

Uważam, że większe zasoby obliczeniowe nie idą w parze z urządzeniami brzegowymi, ale na pewno można by przeprowadzić testy na sprzęcie, by się o tym przekonać. 

Jeżeli chodzi o połączenia resztkowe:

- Poprawiają przepływ gradientu podczas treningu.
- Umożliwia tworzenie głębszych sieci bez problemów z degradacją.
- Lepsza propagacja funkcji w całej sieci.

Sprawdziłem, że można zastosować hybrydowe połączenie tych dwóch rozwiązań. Sprowadza się to do wykorzystania połączeń resztkowych w blokach kodera/dekodera dla stabilności i dodając mechanizm uwagi w strategicznych punktach sieci.

Jak mogłaby wyglądać taka architektura:

1. **Ścieżka enkodera:**
- Bloki splotowe z połączeniami resztkowymi (ResBlocks)
- Każdy blok pomaga zachować przepływ informacji podczas downsamplingu
- Umożliwia głębszą i bardziej stabilną architekturę enkodera
2. **Wąskie gardło:**
- Często obejmuje mechanizm uwagi w celu identyfikacji globalnych relacji
- Pomaga sieci skupić się na najważniejszych cechach przed upsamplingiem
3. **Ścieżka dekodera:**
- Bloki upsamplingu z połączeniami resztkowymi
- Potencjalnie lżejsze mechanizmy uwagi w blokach dekodera
4. **Pomijanie połączeń:**
- Może obejmować bramki uwagi, w których cechy enkodera łączą się z dekoderem
- Bramki uwagi filtrują, które cechy enkodera są istotne na każdym poziomie dekodera

To hybrydowe podejście wykorzystuje uzupełniające się mocne strony obu technik - połączenia resztkowe utrzymują przepływ gradientu, podczas gdy mechanizmy uwagi zwiększają wybór cech. Połączenia resztkowe w całej sieci zapewniają stabilność architektoniczną, a mechanizmy uwagi dodają możliwość skupienia się na określonych obszarach obrazu, w razie potrzeby.

Hybrydowa sieć U-Net wiąże się z wyzwaniami przy wdrożeniu na urządzeniach brzegowych:

- Mechanizmy uwagi są kosztowne obliczeniowo (często jest to złożoność kwadratowa)
- Większe wymagania pamięci do przechowywania map uwagi
- Większe opóźnienie wnioskowania
- Większe zużycie energii

Można sobie z tym poradzić poprzez kwantyzację oraz pruning. Jeśli dokładność ma kluczowe znaczenie, można wytrenować pełny model hybrydowy na wydajnym sprzęcie, a następnie użyć destylacji wiedzy, aby przenieść jego możliwości do mniejszej, przyjaznej dla urządzeń brzegowych, architektury bez uwagi.

Innym rozwiązaniem jest po prostu kompromis między wydajnością a efektywnością, czyli zastosowanie prostrzej architektury z resztkowymi połączeniami.

## Omówienie wyników testów

**Współczynnik Dice'a**

- Poprawa o około 800%
- Współczynnik Dice'a mierzy nakładanie się segmentacji przewidywanych i rzeczywistych
- Wartość 0,72 jest ogólnie uważana za dobrą dla zadań segmentacji medycznej

**IoU**

- Poprawa o 1264%
- IoU jest bardziej rygorystyczną metryką niż Dice, mierzącą stosunek nakładania się do całkowitej powierzchni
- Wartość po wytrenowaniu wskazuje na dobrą jakość segmentacji

**Precyzja**

- Po treningu około 79% pikseli, które Twój model identyfikuje jako część struktury docelowej, jest poprawnych

**Recall**

- Model teraz poprawnie identyfikuje około 67% pikseli, które powinny zostać uwzględnione w segmentacji

Po szkoleniu: Model obecnie rejestruje ponad dwie trzecie wszystkich pikseli guza, co oznacza transformację od „pomijania niemal wszystkiego (9%)” do „znajdowania większości ważnych regionów (67%)”.

Końcowe wskaźniki pokazują dobrze wyważony model, który pozwala uniknąć dwóch typowych pułapek w segmentacji obrazów:

1. **Wysoka precyzja przy bardzo niskim recall** (brak większości nieprawidłowości)
2. **Wysoki recall przy bardzo niskiej precyzji** (zbyt wiele fałszywie pozytywnych wyników)

## Analiza równowagi

Wyższa precyzja w stosunku do przypomnienia sugeruje, że model jest nieco konserwatywny — bardziej prawdopodobne jest, że pominie części struktury (fałszywe wyniki negatywne) niż nieprawidłowo oznaczy tło jako część struktury (fałszywe wyniki pozytywne). Ta równowaga może być odpowiednia w zastosowaniach medycznych, w których fałszywe wyniki pozytywne mogą prowadzić do niepotrzebnych interwencji.

## Co można jeszcze poprawić?

Myślę, że taka architektura stanowi dobrą podstawę i punkt odniesienia do wprowadzania kolejnych usprawnień, o których też wspominałem powyżej. Można jeszcze zmienić metodę normalizacji na GroupNorm, ponieważ obrazy USG mają cechy:

- Dużą zmienność między obrazami: Obrazy USG różnią się znacząco między pacjentami, sprzętem i operatorami. GroupNorm dokonuje normalizacji w obrębie mniejszych grup kanałów, co czyni go odpornym na tę zmienność.
- Szum plamkowy: Charakterystyczny szum plamkowy USG tworzy wzory tekstur, które GroupNorm może pomóc standaryzować bez eliminowania ważnych informacji o teksturze.
- Małe wymiary partii (Batch Sizes): GroupNorm działa spójnie niezależnie od rozmiaru partii, w przeciwieństwie do BatchNorm.
- Granice niskiego kontrastu: GroupNorm pomaga wzmocnić subtelne różnice między tkankami, co jest kluczowe w przypadku USG, gdzie granice mogą być słabo zdefiniowane.
