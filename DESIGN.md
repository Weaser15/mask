## Specification
Create periodic arrays that are represented by a rectangular cell.


## Module 1: Tiling
This module consists of a class that will generate positions and rotations of
shapes to generate multiples of a rectangular unit cell and its extent.

Method call 'get_positions(a1, a2=None, a3=None)' will have as value a N sized iterable of 
containers with 2 elements.
Method call 'get_rotations(a1, a2=None, a3=None)' will have as a value a N sized iterable of angles.
Method call 'get_extent(a1, a2=None, a3=None)' returns the Extent of the tiling.


## Module 2: Extent
This module contains a storage class for the extent of a shape. This consists of
two coords of opposite corners: (x0, y0) and (x1, y1). This represents a
rectangle defined by these points.

Method call buffer(distance) adds distance to each side of the extent.
Method call offset(x, y) offsets the entire Extent by x along x and y along y.
Method call centre(x=0.0, y=0.0) centres the Extent around (x, y)


## Module 3: Shape
This module contains a storage class for the shapes behind the arrays as well
as methods to use them.

Method call 'to_shapely()' returns a shapely Polygon or MultiPolygon 
representation.
Method call 'to_image(out=None, resolution=1.0)' rasters the shape into pixels 
and outputs to a file / iostream if specified. Returns a PIL.Image.Image.
Method call 'plot(ax=None)' creates a matplotlib plot of the geometry.
Method call 'get_extent() returns Extent of the Shape.
Method call 'wrap_to(extent) will cut off the parts that are outside the extent
and apply PBC to them to bring them back inside.

## Module 4: Unit
This module contains user-facing functions that generate Shape instances of
individual magnets.
The functions allow generous ways of specifying the shape parameters.

## Module 5: Array
This module contains user-facing functions that generate Shape instances of 
arrays.
The functions allow generous ways of specifying the array parameters.
Arrays will take one or more Shape as input.


