//
//
//pub trait Expression<S>{
//
//}
//pub trait ExpressionAtom<S> : Expression<S>{
//    fn into_object(self) -> S;
//}
//
//pub trait ExpressionEval<S, I, T> : Expression<S>{
//    fn eval(&self, i: I) -> T;
//}
//
//pub trait BinaryOpExpression<S, I, T> : ExpressionEval<S, I, T>{
//    type Rhs : ExpressionEval<S, I, T>;
//    type Lhs : ExpressionEval<S, I, T>;
//    fn binary_op(&self, t1: T, t2: T) -> T;
//    fn split_src(&self, i: I) -> (I, I);
//    fn rhs(&self) -> &Self::Rhs;
//    fn lhs(&self) -> &Self::Lhs;
//}
//
//impl<E, S, I, T> ExpressionEval<S, I, T> for E where E: BinaryOpExpression<S, I, T>{
//    fn eval(&self, i: I) -> T{
//        let (i1, i2) = self.split_src(i);
//        self.binary_op(self.lhs().eval(i1), self.rhs().eval(i2))
//    }
//}