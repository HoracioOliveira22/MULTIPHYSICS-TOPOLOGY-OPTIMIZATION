lc = 0.004;

Point(1) = {1, 0.00126, 0, lc};
Point(2) = {0.95, 0.00999, 0, lc};
Point(3) = {0.9, 0.01857, 0, lc};
Point(4) = {0.8, 0.03436, 0, lc};
Point(5) = {0.7, 0.04867, 0, lc};
Point(6) = {0.6, 0.06086, 0, lc};
Point(7) = {0.5, 0.07036, 0, lc};
Point(8) = {0.4, 0.07667, 0, lc};
Point(9) = {0.3, 0.0788, 0, lc};
Point(10) = {0.25, 0.07784, 0, lc};
Point(11) = {0.2, 0.07505, 0, lc};
Point(12) = {0.15, 0.06988, 0, lc};
Point(13) = {0.1, 0.06139, 0, lc};
Point(14) = {0.075, 0.05542, 0, lc};
Point(15) = {0.05, 0.04749, 0, lc};
Point(16) = {0.025, 0.03648, 0, lc};
Point(17) = {0.0125, 0.0281, 0, lc};
Point(18) = {0, 0, 0, lc}; // Rotation center
Point(19) = {0.0125, -0.02342, 0, lc};
Point(20) = {0.025, -0.0306, 0, lc};
Point(21) = {0.05, -0.03905, 0, lc};
Point(22) = {0.075, -0.04374, 0, lc};
Point(23) = {0.1, -0.04671, 0, lc};
Point(24) = {0.15, -0.0496, 0, lc};
Point(25) = {0.2, -0.05025, 0, lc};
Point(26) = {0.25, -0.04942, 0, lc};
Point(27) = {0.3, -0.0475, 0, lc};
Point(28) = {0.4, -0.04167, 0, lc};
Point(29) = {0.5, -0.03418, 0, lc};
Point(30) = {0.6, -0.02588, 0, lc};
Point(31) = {0.7, -0.01759, 0, lc};
Point(32) = {0.8, -0.00992, 0, lc};
Point(33) = {0.9, -0.003510, 0, lc};
Point(34) = {0.95, -0.00101, 0, lc};
Point(35) = {1, -0.00126, 0, lc};

// Points for the small rectangle near the trailing edge
Point(36) = {0.1, -0.0175, 0, lc};
Point(37) = {0.15, -0.0175, 0, lc};
Point(38) = {0.15, 0.02, 0, lc};
Point(39) = {0.1, 0.02, 0, lc};

// Points for the far-field boundary
Point(40) = {-1, -1, 0, 0.05};
Point(41) = {-1, 1, 0, 0.05};
Point(42) = {3, 1, 0, 0.05};
Point(43) = {3, -1, 0, 0.05};

// --- Curve Definitions ---
// Airfoil contour (BSpline)
BSpline(1) = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,1};

// Lines for the small rectangle
Line(2) = {39, 38};
Line(3) = {38, 37};
Line(4) = {37, 36};
Line(5) = {36, 39};

// Lines for the far-field boundary
Line(6) = {40, 41};
Line(7) = {41, 42};
Line(8) = {42, 43};
Line(9) = {43, 40};

// --- Apply Rotation ---
// Define rotation parameters
myAngleDeg = -5.0; // Angle of rotation in degrees
myAngleRad = myAngleDeg * Pi / 180.0; // Convert angle to radians

// Axis of rotation: Z-axis {0,0,1}
// Point on the axis of rotation: Origin {0,0,0} (which is Point 18)
Rotate {{0,0,1}, {0,0,0}, myAngleRad } {
  Curve{1}; // Rotate BSpline(1) (the airfoil contour)
  Line{2,3,4,5}; // Rotate the lines forming the small rectangle
}

Curve Loop(100) = {1};
Curve Loop(101) = {2,3,4,5};
Curve Loop(102) = {6,7,8,9};

Plane Surface(200) = {100,101};

Plane Surface(201) = {102,100};


Physical Surface("fluid") = {201};
Physical Surface("solid") = {200};
Physical Curve("inlet") = {6};
Physical Curve("outlet") = {8};
Physical Curve("noSlip") = {7, 9};
Physical Curve("fixed") = {2,3,4,5};
Physical Curve("interface") = {1};
Physical Point("fixed") = {36,37,38,39};
Physical Point("interface") = {1};
Physical Point("noSlip") = {40,41};
Physical Point("outlet") = {42,43};

Mesh 2;

