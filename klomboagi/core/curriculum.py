"""
Teaching Protocol — structured knowledge curricula for Cognitive Genesis.

Instead of teaching one fact at a time through conversation, this module
provides pre-built curricula organized by domain. Each curriculum is a
sequence of (subject, predicate) pairs ordered so foundations come first.

Design principles:
- General truths, not trivia
- Mathematical and logical relationships
- Each fact connects to others in the knowledge graph
- Ordered: foundations before specializations
- Domain-agnostic format: always (subject, predicate)

Usage:
    from klomboagi.core.curriculum import CURRICULA, get_curriculum
    facts = get_curriculum("mathematics")
    for subject, predicate in facts:
        genesis.base._learn_from_teaching_quiet({"subject": subject, "predicate": predicate})
"""

from __future__ import annotations


def get_curriculum(domain: str) -> list[tuple[str, str]]:
    """Get a curriculum by domain name. Returns list of (subject, predicate)."""
    domain = domain.lower().strip()
    if domain in CURRICULA:
        return CURRICULA[domain]
    # Try partial match
    for key in CURRICULA:
        if domain in key or key in domain:
            return CURRICULA[key]
    return []


def get_all_domains() -> list[str]:
    """List all available curriculum domains."""
    return sorted(CURRICULA.keys())


# ═══════════════════════════════════════════════════════════════
# CURRICULA — ordered sequences of (subject, predicate) facts
# ═══════════════════════════════════════════════════════════════

CURRICULA: dict[str, list[tuple[str, str]]] = {}


# ── FOUNDATIONS: Logic & Reasoning ──

CURRICULA["logic"] = [
    # Core concepts
    ("logic", "the study of valid reasoning"),
    ("a proposition", "a statement that is either true or false"),
    ("truth", "a property of propositions that correspond to reality"),
    ("falsity", "a property of propositions that do not correspond to reality"),

    # Logical operations
    ("negation", "the logical opposite of a proposition"),
    ("conjunction", "a compound proposition that is true when both parts are true"),
    ("disjunction", "a compound proposition that is true when at least one part is true"),
    ("implication", "a relation where one proposition necessarily follows from another"),
    ("equivalence", "a relation where two propositions always have the same truth value"),

    # Reasoning types
    ("deduction", "reasoning from general rules to specific conclusions"),
    ("induction", "reasoning from specific observations to general rules"),
    ("abduction", "reasoning from an observation to its most likely explanation"),
    ("analogy", "reasoning by structural similarity between domains"),

    # Logical laws
    ("the law of identity", "a thing is identical to itself"),
    ("the law of non-contradiction", "a proposition cannot be both true and false"),
    ("the law of excluded middle", "a proposition is either true or false with no third option"),
    ("modus ponens", "if P implies Q and P is true then Q must be true"),
    ("modus tollens", "if P implies Q and Q is false then P must be false"),

    # Validity
    ("a valid argument", "an argument where the conclusion follows from the premises"),
    ("a sound argument", "a valid argument with true premises"),
    ("a fallacy", "an error in reasoning that makes an argument invalid"),
    ("a contradiction", "a proposition that is always false"),
    ("a tautology", "a proposition that is always true"),
]


# ── MATHEMATICS: Numbers & Arithmetic ──

CURRICULA["mathematics"] = [
    # Foundations
    ("mathematics", "the study of patterns, quantities, structures, and relationships"),
    ("a number", "an abstract concept representing quantity"),
    ("zero", "the number representing nothing or the absence of quantity"),
    ("one", "the first counting number and the multiplicative identity"),
    ("infinity", "a concept representing a quantity without bound"),

    # Number types
    ("a natural number", "a positive counting number starting from 1"),
    ("an integer", "a whole number that can be positive, negative, or zero"),
    ("a rational number", "a number expressible as a fraction of two integers"),
    ("an irrational number", "a number that cannot be expressed as a fraction"),
    ("a real number", "any number on the continuous number line"),
    ("a complex number", "a number with both a real part and an imaginary part"),
    ("a prime number", "a natural number greater than 1 divisible only by 1 and itself"),
    ("pi", "the ratio of a circle's circumference to its diameter, approximately 3.14159"),
    ("euler's number", "the base of natural logarithms, approximately 2.71828"),

    # Operations
    ("addition", "the operation of combining two quantities"),
    ("subtraction", "the operation of finding the difference between two quantities"),
    ("multiplication", "the operation of repeated addition"),
    ("division", "the operation of splitting a quantity into equal parts"),
    ("exponentiation", "the operation of multiplying a number by itself repeatedly"),
    ("a square root", "a number that when multiplied by itself gives the original"),

    # Properties
    ("commutativity", "the property where order does not affect the result"),
    ("associativity", "the property where grouping does not affect the result"),
    ("distributivity", "the property where multiplication distributes over addition"),
    ("the additive identity", "zero, because adding zero to any number gives that number"),
    ("the multiplicative identity", "one, because multiplying any number by one gives that number"),

    # Relations
    ("equality", "a relation meaning two expressions have the same value"),
    ("inequality", "a relation meaning two expressions have different values"),
    ("greater than", "a relation meaning one quantity exceeds another"),
    ("less than", "a relation meaning one quantity is smaller than another"),

    # Structures
    ("a set", "a collection of distinct objects"),
    ("a function", "a relation that maps each input to exactly one output"),
    ("a variable", "a symbol representing an unknown or changing quantity"),
    ("an equation", "a statement that two expressions are equal"),
    ("a proof", "a logical argument establishing the truth of a statement"),
    ("an axiom", "a statement accepted as true without proof"),
    ("a theorem", "a statement proven true from axioms and other theorems"),
]


# ── GEOMETRY ──

CURRICULA["geometry"] = [
    # Foundations
    ("geometry", "the branch of mathematics studying shapes, sizes, and spatial relationships"),
    ("a point", "a location in space with no size"),
    ("a line", "a one-dimensional figure extending infinitely in both directions"),
    ("a plane", "a flat two-dimensional surface extending infinitely"),
    ("space", "the three-dimensional extent in which objects exist"),
    ("a dimension", "an independent direction of measurement"),

    # Basic shapes
    ("an angle", "the figure formed by two rays sharing an endpoint"),
    ("a right angle", "an angle measuring exactly 90 degrees"),
    ("a triangle", "a polygon with three sides and three angles"),
    ("a rectangle", "a quadrilateral with four right angles"),
    ("a square", "a rectangle with all sides equal"),
    ("a circle", "the set of all points equidistant from a center point"),
    ("a polygon", "a closed figure with straight sides"),
    ("a quadrilateral", "a polygon with four sides"),

    # Properties
    ("perimeter", "the total distance around the boundary of a shape"),
    ("area", "the measure of space enclosed by a two-dimensional shape"),
    ("volume", "the measure of space enclosed by a three-dimensional shape"),
    ("circumference", "the perimeter of a circle, equal to 2 times pi times radius"),
    ("radius", "the distance from the center of a circle to its edge"),
    ("diameter", "the distance across a circle through its center, twice the radius"),

    # Triangle properties
    ("the pythagorean theorem", "in a right triangle, a squared plus b squared equals c squared"),
    ("a right triangle", "a triangle with one 90-degree angle"),
    ("an equilateral triangle", "a triangle with all sides equal and all angles 60 degrees"),
    ("an isosceles triangle", "a triangle with two sides equal"),
    ("the sum of angles in a triangle", "always 180 degrees"),

    # 3D shapes
    ("a sphere", "the set of all points equidistant from a center in three dimensions"),
    ("a cube", "a solid with six equal square faces"),
    ("a cylinder", "a solid with two parallel circular bases connected by a curved surface"),
    ("a cone", "a solid that narrows from a circular base to a point"),

    # Transformations
    ("symmetry", "a property where a shape looks the same after a transformation"),
    ("rotation", "a transformation that turns a shape around a point"),
    ("reflection", "a transformation that mirrors a shape across a line"),
    ("translation", "a transformation that slides a shape without rotating"),
    ("congruence", "two shapes having the same size and shape"),
    ("similarity", "two shapes having the same shape but possibly different sizes"),

    # Coordinate geometry
    ("the cartesian plane", "a two-dimensional coordinate system using x and y axes"),
    ("the origin", "the point where coordinate axes intersect, at position zero zero"),
    ("slope", "the measure of steepness of a line, rise over run"),
    ("distance formula", "derived from the pythagorean theorem for coordinate geometry"),
]


# ── PHYSICS ──

CURRICULA["physics"] = [
    # Foundations
    ("physics", "the study of matter, energy, and the fundamental forces of nature"),
    ("matter", "anything that has mass and occupies space"),
    ("energy", "the capacity to do work or cause change"),
    ("a force", "an interaction that changes the motion of an object"),
    ("mass", "a measure of the amount of matter in an object"),
    ("time", "the dimension in which events occur in sequence"),

    # Mechanics
    ("velocity", "the rate of change of position with respect to time"),
    ("acceleration", "the rate of change of velocity with respect to time"),
    ("momentum", "mass multiplied by velocity"),
    ("inertia", "the tendency of an object to resist changes in its state of motion"),
    ("newton's first law", "an object at rest stays at rest unless acted on by a force"),
    ("newton's second law", "force equals mass times acceleration"),
    ("newton's third law", "every action has an equal and opposite reaction"),
    ("gravity", "a fundamental force of attraction between objects with mass"),
    ("friction", "a force that opposes relative motion between surfaces"),
    ("work", "force applied over a distance, measured in joules"),
    ("power", "the rate of doing work, measured in watts"),
    ("kinetic energy", "the energy an object has due to its motion"),
    ("potential energy", "stored energy due to position or configuration"),
    ("conservation of energy", "energy cannot be created or destroyed, only transformed"),
    ("conservation of momentum", "total momentum in a closed system remains constant"),

    # Thermodynamics
    ("temperature", "a measure of the average kinetic energy of particles"),
    ("heat", "the transfer of thermal energy between objects"),
    ("entropy", "a measure of disorder in a system, always increases overall"),
    ("the first law of thermodynamics", "energy is conserved in all processes"),
    ("the second law of thermodynamics", "entropy of an isolated system never decreases"),
    ("absolute zero", "the lowest possible temperature where particle motion stops"),

    # Electromagnetism
    ("electric charge", "a fundamental property of matter, positive or negative"),
    ("an electric field", "a region where electric charges experience force"),
    ("a magnetic field", "a region where magnetic materials experience force"),
    ("an electromagnetic wave", "a wave of oscillating electric and magnetic fields"),
    ("light", "an electromagnetic wave visible to humans"),
    ("the speed of light", "approximately 299,792,458 meters per second in vacuum"),

    # Waves
    ("a wave", "a disturbance that transfers energy through a medium"),
    ("wavelength", "the distance between successive peaks of a wave"),
    ("frequency", "the number of wave cycles per unit of time"),
    ("amplitude", "the maximum displacement of a wave from equilibrium"),

    # Modern physics
    ("relativity", "einstein's theory relating space, time, mass, and energy"),
    ("quantum mechanics", "the physics of particles at atomic and subatomic scales"),
    ("the photoelectric effect", "light striking metal ejects electrons, proving light has particle nature"),
    ("wave-particle duality", "all matter exhibits both wave and particle properties"),
    ("the uncertainty principle", "position and momentum cannot both be known precisely"),
    ("superposition", "a quantum system can exist in multiple states simultaneously"),
    ("entanglement", "quantum particles linked so measuring one affects the other instantly"),
]


# ── ECONOMICS ──

CURRICULA["economics"] = [
    # Foundations
    ("economics", "the study of how societies allocate scarce resources"),
    ("scarcity", "the condition where wants exceed available resources"),
    ("a market", "where buyers and sellers exchange goods and services"),
    ("supply", "the quantity of a good producers are willing to sell at a given price"),
    ("demand", "the quantity of a good consumers are willing to buy at a given price"),
    ("equilibrium", "the point where supply equals demand"),
    ("price", "the amount of money required to purchase a good or service"),

    # Key concepts
    ("opportunity cost", "the value of the next best alternative given up"),
    ("marginal cost", "the cost of producing one additional unit"),
    ("marginal utility", "the additional satisfaction from consuming one more unit"),
    ("diminishing returns", "each additional input yields less additional output"),
    ("comparative advantage", "producing a good at a lower opportunity cost than others"),
    ("absolute advantage", "producing more of a good with the same resources"),
    ("trade", "the voluntary exchange of goods and services between parties"),

    # Macro
    ("gdp", "the total value of goods and services produced in a country"),
    ("inflation", "a general increase in prices over time"),
    ("deflation", "a general decrease in prices over time"),
    ("unemployment", "the condition where people seeking work cannot find it"),
    ("a recession", "a significant decline in economic activity lasting months"),
    ("fiscal policy", "government use of taxation and spending to influence the economy"),
    ("monetary policy", "central bank actions to control money supply and interest rates"),
    ("interest rate", "the cost of borrowing money, expressed as a percentage"),

    # Markets
    ("perfect competition", "a market with many sellers offering identical products"),
    ("a monopoly", "a market with only one seller"),
    ("an oligopoly", "a market dominated by a few large sellers"),
    ("externality", "a cost or benefit affecting parties not involved in a transaction"),
    ("public good", "a good that is non-excludable and non-rivalrous"),
    ("market failure", "when free markets fail to allocate resources efficiently"),
]


# ── COMPUTER SCIENCE ──

CURRICULA["computer science"] = [
    # Foundations
    ("computer science", "the study of computation, information, and automation"),
    ("an algorithm", "a finite sequence of well-defined instructions to solve a problem"),
    ("data", "information encoded in a form suitable for processing"),
    ("a program", "a set of instructions that tells a computer what to do"),
    ("computation", "the process of calculating or processing information"),

    # Data structures
    ("a data structure", "a way of organizing data for efficient access and modification"),
    ("an array", "a collection of elements stored at contiguous memory positions"),
    ("a linked list", "a sequence of elements where each points to the next"),
    ("a stack", "a collection where the last element added is the first removed"),
    ("a queue", "a collection where the first element added is the first removed"),
    ("a tree", "a hierarchical data structure with a root and children"),
    ("a graph", "a data structure of nodes connected by edges"),
    ("a hash table", "a structure that maps keys to values using a hash function"),

    # Algorithms
    ("time complexity", "how the running time of an algorithm grows with input size"),
    ("space complexity", "how the memory usage of an algorithm grows with input size"),
    ("big-o notation", "a way to describe the upper bound of an algorithm's growth rate"),
    ("sorting", "the process of arranging elements in a specific order"),
    ("searching", "the process of finding a specific element in a collection"),
    ("recursion", "a technique where a function calls itself to solve subproblems"),
    ("dynamic programming", "solving complex problems by breaking them into overlapping subproblems"),

    # Computation theory
    ("a turing machine", "a theoretical model of computation that can simulate any algorithm"),
    ("computability", "whether a problem can be solved by any algorithm"),
    ("the halting problem", "the undecidable problem of whether a program will finish running"),
    ("np-completeness", "a class of problems for which no efficient solution is known"),

    # Systems
    ("binary", "a number system using only 0 and 1"),
    ("a bit", "the smallest unit of information, either 0 or 1"),
    ("a byte", "a group of 8 bits"),
    ("memory", "hardware that stores data and instructions for the processor"),
    ("a processor", "hardware that executes instructions"),
    ("an operating system", "software that manages hardware and provides services to programs"),
    ("a network", "a system of interconnected computers that share resources"),
    ("encryption", "the process of encoding information so only authorized parties can read it"),
]


# ── CHEMISTRY ──

CURRICULA["chemistry"] = [
    ("chemistry", "the study of matter, its properties, and how it changes"),
    ("an atom", "the smallest unit of a chemical element"),
    ("a molecule", "two or more atoms bonded together"),
    ("an element", "a substance made of only one type of atom"),
    ("a compound", "a substance made of two or more different elements bonded together"),
    ("a chemical bond", "an attraction between atoms that holds them together"),
    ("a covalent bond", "a bond where atoms share electrons"),
    ("an ionic bond", "a bond formed by the transfer of electrons between atoms"),
    ("the periodic table", "an arrangement of elements by atomic number and properties"),
    ("atomic number", "the number of protons in an atom's nucleus"),
    ("a proton", "a positively charged particle in the atomic nucleus"),
    ("a neutron", "a neutral particle in the atomic nucleus"),
    ("an electron", "a negatively charged particle orbiting the nucleus"),
    ("an ion", "an atom with a net electric charge from gaining or losing electrons"),
    ("a chemical reaction", "a process that transforms one set of substances into another"),
    ("a catalyst", "a substance that speeds up a reaction without being consumed"),
    ("an acid", "a substance that donates hydrogen ions in solution"),
    ("a base", "a substance that accepts hydrogen ions in solution"),
    ("ph", "a scale measuring how acidic or basic a solution is, from 0 to 14"),
    ("oxidation", "the loss of electrons by a substance"),
    ("reduction", "the gain of electrons by a substance"),
    ("the law of conservation of mass", "mass is neither created nor destroyed in a reaction"),
]


# ── BIOLOGY ──

CURRICULA["biology"] = [
    ("biology", "the study of living organisms and their processes"),
    ("a cell", "the basic structural and functional unit of all living organisms"),
    ("dna", "the molecule that carries genetic instructions for life"),
    ("a gene", "a segment of dna that codes for a specific protein"),
    ("a protein", "a molecule made of amino acids that performs functions in cells"),
    ("evolution", "the change in inherited characteristics of populations over generations"),
    ("natural selection", "the process where organisms with favorable traits survive more"),
    ("a species", "a group of organisms capable of interbreeding"),
    ("photosynthesis", "the process by which plants convert sunlight into chemical energy"),
    ("cellular respiration", "the process of converting glucose and oxygen into energy"),
    ("an ecosystem", "a community of organisms interacting with their environment"),
    ("homeostasis", "the maintenance of stable internal conditions in an organism"),
    ("mitosis", "cell division producing two identical daughter cells"),
    ("meiosis", "cell division producing four cells with half the chromosome number"),
    ("a mutation", "a change in the dna sequence of an organism"),
    ("taxonomy", "the science of classifying organisms into groups"),
    ("a virus", "a microscopic agent that replicates only inside living cells"),
    ("a bacterium", "a single-celled organism without a nucleus"),
    ("symbiosis", "a close relationship between two different species"),
]


# ── CATEGORY THEORY (for conflict detection) ──

CURRICULA["categories"] = [
    # Colors
    ("red", "a color"), ("blue", "a color"), ("green", "a color"),
    ("yellow", "a color"), ("orange", "a color"), ("purple", "a color"),
    ("black", "a color"), ("white", "a color"), ("pink", "a color"),
    ("brown", "a color"),

    # Sizes
    ("big", "a size"), ("small", "a size"), ("large", "a size"),
    ("tiny", "a size"), ("huge", "a size"), ("little", "a size"),

    # Temperatures
    ("hot", "a temperature"), ("cold", "a temperature"),
    ("warm", "a temperature"), ("cool", "a temperature"),
    ("freezing", "a temperature"), ("boiling", "a temperature"),

    # States of matter
    ("solid", "a state of matter"), ("liquid", "a state of matter"),
    ("gas", "a state of matter"), ("plasma", "a state of matter"),

    # Animal classes
    ("mammal", "an animal class"), ("reptile", "an animal class"),
    ("bird", "an animal class"), ("fish", "an animal class"),
    ("insect", "an animal class"), ("amphibian", "an animal class"),
    ("arachnid", "an animal class"),

    # Speeds
    ("fast", "a speed"), ("slow", "a speed"), ("quick", "a speed"),
]


# ── Totals ──

def curriculum_stats() -> dict:
    """Stats about all curricula."""
    return {
        "domains": len(CURRICULA),
        "total_facts": sum(len(c) for c in CURRICULA.values()),
        "per_domain": {k: len(v) for k, v in CURRICULA.items()},
    }
