CXX = g++
CXXFLAGS = -Wall -O
TARGET = sim
SRC = thermo.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC) -lm

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET) report_*
