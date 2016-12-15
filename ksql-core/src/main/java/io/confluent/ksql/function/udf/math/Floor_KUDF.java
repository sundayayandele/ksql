package io.confluent.ksql.function.udf.math;


import io.confluent.ksql.function.KSQLFunctionException;
import io.confluent.ksql.function.udf.KUDF;

public class Floor_KUDF implements KUDF {
  @Override
  public void init() {
  }

  @Override
  public Object evaluate(Object... args) {
    if (args.length != 1) {
      throw new KSQLFunctionException("Floor udf should have one input argument.");
    }
    return Math.floor((Double) args[0]);
  }
}