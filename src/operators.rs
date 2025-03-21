use crate::tensor::Tensor;
use rayon::prelude::*;

// get (row) vectors from a 2D table given a list of indices
pub fn gather(y: &mut Tensor<f32>, indices: &Tensor<u32>, table: &Tensor<f32>) {
    let length = indices.size();//我要提取几行，也就是提取几个向量（二维）
    let table_shape = table.shape();//这里的shape是自定义（非rust），就是结构体中的一个值。这里二维应该返回的是[8,9]这样的
    assert!(table_shape.len() == 2);
    let dim = table_shape[1];//获取每行向量有几个分量，二维的话就是列数
    assert!(y.size() == length * dim);
    for i in 0..length {
        let src = &table.data()[indices.data()[i] as usize * dim..][..dim];
        let dst = &mut unsafe { y.data_mut() }[i * dim..][..dim];
        dst.copy_from_slice(src);
    }
}

// RoPE: Rotary Positional Embedding
pub fn rope(y: &mut Tensor<f32>, start_pos: usize, theta: f32) {//2
    let shape = y.shape();
    assert!(shape.len() == 3);
    let seq_len = shape[0];
    let n_heads = shape[1];
    let d = shape[2];
    let data = unsafe { y.data_mut() };
    for tok in 0..seq_len {
        let pos = start_pos + tok;
        for head in 0..n_heads {
            for i in 0..d / 2 {
                let a = data[tok * n_heads * d + head * d + i];
                let b = data[tok * n_heads * d + head * d + i + d / 2];
                let freq = pos as f32 / theta.powf((i * 2) as f32 / d as f32);
                let (sin, cos) = freq.sin_cos();
                data[tok * n_heads * d + head * d + i] = a * cos - b * sin;
                data[tok * n_heads * d + head * d + i + d / 2] = b * cos + a * sin;
            }
        }
    }
}

// softmax(x) = exp(x - max) / sum(exp(x - max))
// y = softmax(mask(x))
pub fn masked_softmax(y: &mut Tensor<f32>) {//3
    let ndim = y.shape().len();
    assert!(ndim >= 2);
    let seq_len = y.shape()[ndim - 2];
    let total_seq_len = y.shape()[ndim - 1];
    let batch = y.size() / (seq_len * total_seq_len);
    let data = unsafe { y.data_mut() };
    for b in 0..batch {
        let base = b * seq_len * total_seq_len;
        for i in 0..seq_len {
            let offset = base + i * total_seq_len;
            let boundary = total_seq_len - seq_len + i + 1;

            let max = data[offset..offset + boundary]
                .iter()
                .fold(data[offset], |a, b| a.max(*b));

            let sum = (0..boundary)
                .map(|j| {
                    let e = (data[offset + j] - max).exp();
                    data[offset + j] = e;
                    e
                })
                .sum::<f32>();

            (0..boundary).for_each(|j| data[offset + j] /= sum);
            (boundary..total_seq_len).for_each(|j| data[offset + j] = 0.0);
        }
    }
}

pub fn rms_norm(y: &mut Tensor<f32>, x: &Tensor<f32>, w: &Tensor<f32>, epsilon: f32) {//传参三个张量：输入、输出、权重张量 epsilon：一个很小的值，用于避免除零错误，常见的做法是在计算 RMS 时加上一个微小的常数来稳定计算。
    // todo!("实现 rms_norm，计算前做一些必要的检查会帮助你后续调试");//均方根计算逻辑：对当前行的元素平方求和、除以元素个数，再加上 epsilon 后取平方根
    assert!(w.size()==x.shape().last().copied().unwrap_or(0),"w向量的分量个数和x向量的分量个数不一样");//因为w是一维的，所以size返回的才是向量的分量个数
    assert!(w.shape().len()==1,"w不是一维向量");
    assert!(x.shape().last().copied().unwrap_or(0)==y.shape().last().copied().unwrap_or(0),"x，y向量分量长度一样");

    let x_data=x.data();
    let w_data=w.data();//获取到底层一维长数据

    let vec_num=x.shape().last().copied().unwrap_or(0);//获取一维向量的分量个数
    
    unsafe {

        y.data_mut().par_chunks_mut(vec_num).zip(x_data.par_chunks(vec_num))//按照分量个数划分向量获取所有的向量-zip将xy合成一个元组后面遍历的时候方便操作，zip是在原内存空间上操作的
        .for_each(|(y_slice,x_slice)|{//y_slice和x_slice这个代表张量xy的每一个长度为n的向量 
            let rms = (x_slice.iter().map(|&val| val*val).sum::<f32>()/vec_num as f32 +epsilon).sqrt();//分母部分

            for i in 0..vec_num{
                y_slice[i]=w_data[i]*x_slice[i]/rms;//分子部分
            }
        });

    }

 
}

// y = silu(x) * y
// hint: this is an element-wise operation
pub fn swiglu(y: &mut Tensor<f32>, x: &Tensor<f32>) {//5
    // let len = y.size();
    // assert!(len == x.size());

    // let _y = unsafe { y.data_mut() };
    // let _x = x.data();

    // todo!("实现 silu，这里给了一些前期准备工作的提示，你可以参考");

    // x.data()就是获取Tensor里的数据，如果Tensor是[1,2,3,4,5,6]，那么let a = x.data()
    // a的内容就是[1,2,3,4,5,6]
    let length=y.size();
    let len=y.size();
    assert!(len==x.size());
    // assert!(x.size()==y.size());
    let y_data = unsafe { y.data_mut() };
    let x_data = x.data();
    for i in 0..length{
        y_data[i]=y_data[i]*x_data[i]/(1.0+(-x_data[i]).exp());
    }
}

// C = beta * C + alpha * A @ B^T
// hint: You don't need to do an explicit transpose of B
pub fn matmul_transb(c: &mut Tensor<f32>, beta: f32, a: &Tensor<f32>, b: &Tensor<f32>, alpha: f32) {//6
    // operators.rs中的matmul_transb

    // todo!("实现 matmul_transb，计算前做一些必要的检查会帮助你后续调试");
    let (a_row, a_col) = (a.shape()[0], a.shape()[1]);
    let (b_row, b_col) = (b.shape()[0], b.shape()[1]);
    let (c_row, c_col) = (c.shape()[0], c.shape()[1]);

    assert!(a_col == b_col, "Inner dimensions of A and B must match");
    assert!(
        a_row == c_row && b_row == c_col,
        "Output matrix C must have shape (a_row, b_row)"
    );

    let a_data = a.data();
    let b_data = b.data();
    let c_data = unsafe { c.data_mut() };

    for i in 0..c_row {
        for j in 0..c_col {
            let mut sum = 0.0;
            for k in 0..a_col {
                sum += a_data[i * a_col + k] * b_data[j * b_col + k];
            }
            c_data[i * c_col + j] = beta * c_data[i * c_col + j] + alpha * sum;
        }
    }
}

// Dot product of two tensors (treated as vectors)
#[allow(unused)]
pub fn dot(x: &Tensor<f32>, y: &Tensor<f32>) -> f32 {//7
    let len = x.size();
    assert!(len == y.size());
    let x_ = x.data();
    let y_ = y.data();
    let mut sum = 0.0;
    for i in 0..len {
        sum += x_[i] * y_[i];
    }
    sum
}

// Sample a index from a tensor (treated as a probability vector)
pub fn random_sample(x: &Tensor<f32>, top_p: f32, top_k: u32, temperature: f32) -> u32 {//8
    assert!(x.shape()[x.shape().len() - 1] == x.size());
    if temperature <= 0. || top_k < 2 || top_p <= 0. {
        return x
            .data()
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0 as _;
    }

    #[derive(Clone, Copy, PartialEq, Debug)]
    struct Probability {
        val: f32,
        tok: u32,
    }
    impl Eq for Probability {}
    impl PartialOrd for Probability {
        #[inline]
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(other))
        }
    }
    impl Ord for Probability {
        #[inline]
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            match self.val.total_cmp(&other.val) {
                std::cmp::Ordering::Equal => self.tok.cmp(&other.tok),
                ord => ord.reverse(),
            }
        }
    }
    impl From<(usize, &f32)> for Probability {
        #[inline]
        fn from((i, p): (usize, &f32)) -> Self {
            Self {
                val: p.clone(),
                tok: i as _,
            }
        }
    }

    // sort
    let mut logits = x
        .data()
        .iter()
        .enumerate()
        .map(Probability::from)
        .collect::<Vec<_>>();
    logits.sort_unstable();
    let max = core::mem::replace(&mut logits[0].val, 1.);
    // softmax & sum
    for i in 1..logits.len() {
        logits[i].val = logits[i - 1].val + ((logits[i].val - max) / temperature).exp();
    }
    // topk & topp & random
    let pk = logits[(top_k as usize).min(logits.len()) - 1].val;
    let pp = logits[logits.len() - 1].val * top_p;
    let plimit = rand::random::<f32>() * f32::min(pk, pp);
    // sample
    logits.iter().find(|p| p.val >= plimit).unwrap().tok
}

// Your implementation should at least pass the following tests:
#[test]
fn test_silu() {//
    let mut y = Tensor::<f32>::new(vec![2., 3., 4.], &vec![1, 3]);
    let x = Tensor::<f32>::new(vec![1., 2., 3.], &vec![1, 3]);
    swiglu(&mut y, &x);
    assert!(y.close_to(
        &Tensor::<f32>::new(vec![1.4621172, 5.2847824, 11.43089], &vec![1, 3]),
        1e-3
    ));
}

#[test]
fn test_rms_norm() {//10
    let mut y = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
    let x = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
    let w = Tensor::<f32>::new(vec![1., 2.], &vec![2]);
    rms_norm(&mut y, &x, &w, 1e-6);
    assert!(y.close_to(
        &Tensor::<f32>::new(
            vec![0.6324554, 2.5298216, 0.8485281, 2.2627416],
            &vec![2, 2]
        ),
        1e-3
    ));
}

#[test]
fn test_matmul_transb() {//11
    let mut c = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
    let a = Tensor::<f32>::new(vec![1., 2., 3., 4., 5., 6.], &vec![2, 3]);
    let b = Tensor::<f32>::new(vec![1., 2., 3., 4., 5., 6.], &vec![2, 3]);
    matmul_transb(&mut c, 1., &a, &b, 1.);
    assert!(c.close_to(
        &Tensor::<f32>::new(vec![15., 34., 35., 81.], &vec![2, 2]),
        1e-3
    ));
}
