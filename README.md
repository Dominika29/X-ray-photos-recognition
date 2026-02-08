


  <h3 align="center">x-ray photo recognition</h3>

  <p align="center">
	  Dominika Czerwińska 159557
	</p>
  <p align="center">
	  Bartosz Cywiński 159467
	</p>
	  <p align="center">
   Convolutional Neural Network
   </p>



## Table of contents

- [Opis Projektu](#opis-projektu)
- [Metody rozwiązania problemu](#metody-rozwiązania-problemu)
- [Opis Wybranej Koncepcji](#opis-wybranej-koncepcji)
- [Proof of concept](#proof-of-concept)


## Opis Projektu
Problemem, na którym oparliśmy nasz projekt jest rozpoznawanie złamań kończyn górnych oraz dolnych. Rozpoznanie przebiega poprzez ocenę zdjęcia rentgenowskiego. Do rozwiązania problemu użyliśmy konwolucyjnej sieci neuronowej (CNN - Convolutional Neural Network). Konwolucyjne sieci neuronowe to rodzaj algorytmów głębokiego uczenia, które wykorzystują operację splotu do automatycznego wykrywania i wyodrębniania istotnych cech z danych o strukturze siatki, takich jak obrazy. Dzięki specyficznej architekturze potrafią one rozpoznawać wzorce wizualne (np. krawędzie, kształty czy obiekty) przy zachowaniu niezmienności względem ich przesunięcia w przestrzeni.

- ### Wejście 
  Zdjęcia rentgenowskie (X-ray) kończyn górnych oraz dolnych. Każde zdjęcie jest standaryzowane do odcieni szarości o jednolitej rozdzielczości (224x224).
  Model klasyfikuje zdjęcia do dwóch kategorii: złamanie (fractured) lub nie-złamanie (not fractured).
  
- ### Wyjście
  Poprawna diagnostyka prostych złamań, poprzez odpowiednio dobrany opis zdjęć rentgenowskich.
  
- ### Motywacja 
  Technologia ta znacznie usprawniłaby przepływ pracy w placówkach medycznych, stanowiłaby wsparcie dla radiologów w warunkach wysokiej
  presji oraz zminimalizowała odsetek „przeoczeń” diagnostycznych zwłaszcza na oddziałach typu SOR.
  
## Metody rozwiązania problemu
| **Metodologia uczenia maszynowego** | **Opis ogólny** | **Zalety** | **Wady** |
|:---:|:---:|:---:|:---:|
| **CNN** (Konwolucyjne sieci neuronowe) | Wyspecjalizowane warstwy automatycznie identyfikują hierarchie przestrzenne i wzorce (np. pęknięcia kości) bezpośrednio z surowych pikseli. | Najwyższa dokładność w obrazowaniu medycznym <br>Automatyczne wyodrębnianie cech <br> Wysoka wydajność dzięki współdzieleniu wag | Duże zapotrzebowanie na zasoby GPU <br>Wydajność uzależniona od jakości obrazu |
| **DNN** (Głębokie sieci neuronowe) | Składają się z wielu gęstych, w pełni połączonych warstw, które mapują złożone nieliniowe zależności w danych. | Wszechstronność dla różnych typów danych <br>Możliwość budowania wysoce złożonych modeli | gnorują przestrzenne relacje między pikselami <br> Ekstremalnie czaso- i danochłonne <br> Dłuższy czas trenowania|
| **Tradycyjna ekstrakcja cech** | Ręcznie zdefiniowane deskryptory (np. gęstość kości, ostrość krawędzi) są wprowadzane do klasyfikatora płytkiego. | Niskie zapotrzebowanie na moc obliczeniową <br> Wysoka przejrzystość i interpretowalność <br>Dobrze sprawdza się przy ograniczonych zbiorach danych |Dokładność ograniczona przez dobór cech przez człowieka <br> Nie wychwytuje subtelnych niuansów wizualnych |

## Opis Wybranej Koncepcji
Wybranym rozwiązaniem jest konwolucyjna sieć neuronowa (CNN) wykorzystująca architekturę ResNet-50 oraz technikę Transfer Learningu. 
Podstawowym mechanizmem CNN jest stosowanie różnych jąder (filtrów) do danych wejściowych w celu zidentyfikowania markerów diagnostycznych, takich jak przerwania ciągłości kości czy zaburzenia osiowości.

Zastosowanie tych sieci składa się z następujących etapów:4

- ###	Warstwy splotowe  i Pooling
	Wykorzystano strukturę ResNet-50, która automatycznie ekstraktuje cechy wizualne. Warstwy splotowe identyfikują krawędzie i cienie,
	a warstwy typu pooling redukują wymiarowość, zachowując kluczowe informacje o strukturze kości.
	
- ###	Warstwy gęste (Dense Layers)
	Oryginalny klasyfikator sieci został zastąpiony warstwą torch.nn.Linear(num_filters, 1).
	Jest to warstwa w pełni połączona, która agreguje cechy wypracowane przez model ResNet i sprowadza je do jednej wartości decyzyjnej.

- ###   Wyjście Sigmoid (zamiast Softmax)
	Ponieważ rozwiązujemy problem klasyfikacji binarnej (Fractured vs Not Fractured), na wyjściu sieci zastosowano funkcję aktywacji Sigmoid.
	Mapuje ona wynik sieci na wartość z przedziału $(0, 1)$, reprezentującą prawdopodobieństwo wystąpienia klasy pozytywnej.

	# Dane
	Wykorzystany został publicznie dostępny zbiór danych (FracAtlas Dataset - 4,083 images which have been manually annotated for classification, localization and segmentation of bone fractures with the help of 2 expert radiologists and later validated by a medical officer), zawierający zdjęcia RTG.

	# Wynik systemu
	Algorytm generuje skalar prawdopodobieństwa $P$. Interpretacja wyniku odbywa się następująco:
	•	Jeśli $P > 0.5$ (próg odcięcia), system klasyfikuje obraz jako złamanie.
	•	Wektor prawdopodobieństwa jest wyliczany jako $[P, 1-P]$. Przykładowo, wynik 0.915 oznacza 91,5% szans na złamanie oraz 8,5% szans na brak złamania

  # Procedura testowania
	W kodzie zaimplementowaliśmy metryki: accuracy_score, precision_score, recall_score, f1_score oraz macierz pomyłek (Confusion Matrix). 
System oblicza:
•  Recall (Czułość): informuje, jaki procent faktycznych złamań został wykryty.
•  Precision (Precyzja): mówi o tym, jak często diagnoza złamania jest trafna.
•  F1-Score: Średnia harmoniczna precyzji i czułości, dająca pełny obraz stabilności modelu.
•  AUC-ROC: Określa zdolność modelu do rozróżniania obu klas niezależnie od progu decyzyjnego.
Model jest oceniany na oddzielnym zbiorze danych (dl_test), którego nie widział podczas procesu uczenia. Pozwala to sprawdzić zdolność algorytmu do generalizacji wiedzy na nowe przypadki.

  # Napotkane Problemy
 	• Jakość danych wejściowych: Model opiera się na obrazach o rozdzielczości $224 \times 224$ pikseli. W rzeczywistości medycznej zdjęcia RTG mają znacznie wyższą rozdzielczość; proces kompresji może prowadzić do utraty informacji o mikropęknięciach.
		• Zróżnicowanie sprzętowe: Zdjęcia pochodzące z różnych aparatów RTG mogą różnić się kontrastem i poziomem szumów, co wymaga zastosowania zaawansowanej augmentacji danych (w kodzie użyto ColorJitter oraz RandomRotation).
		• Dodaliśmy również balansowanie klas, z powodu nieproporcjonalnej liczby zdjęć kończyn złamanych do zdrowych - funkcja „balance_dataset” rozwiązuje ten problem i pozwala nam uniknąć stronniczości modelu.

## Proof of concept
For this project PyTorch library was chosen.

Steps:
 - ##Training the model:

